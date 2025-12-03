import torch
from torch import nn
from torch.nn import functional as F
import math

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 adapter_id=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.adapter_id = adapter_id
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        output = self.up_proj(down)
        return output
    

class NestedLoRAAdapter(nn.Module):
    """
    Nested LoRA: fast と slow の 2 つの LoRA を並列に持つ。
    Forward 時は両方の出力を足し合わせる。
    """
    def __init__(self, config, adapter_id, dropout=0.0):
        super().__init__()
        self.config = config
        self.adapter_id = adapter_id
        rank = config.nested_lora_rank
        
        # slow LoRA: 長期共通メモリ
        self.slow = Adapter(
            config=config,
            adapter_id=f"{adapter_id}_slow",
            bottleneck=rank,
            dropout=dropout,
            init_option="lora",
            adapter_scalar="1.0",
            adapter_layernorm_option="none"
        )
        
        # fast LoRA: 短期エピソード記憶 (MoE-like)
        # K = nb_tasks (config.nb_tasks)
        self.nb_tasks = getattr(config, "nb_tasks", 1)
        self.fast_adapters = nn.ModuleList([
            Adapter(
                config=config,
                adapter_id=f"{adapter_id}_fast_{i}",
                bottleneck=rank,
                dropout=dropout,
                init_option="lora",
                adapter_scalar="1.0",
                adapter_layernorm_option="none"
            ) for i in range(self.nb_tasks)
        ])
    
    def forward(self, x, task_id=None):
        slow_out = self.slow(x)
        
        if self.training:
            # Training: use specific fast adapter for the current task
            if task_id is None:
                # Fallback if task_id not provided (should not happen in correct usage)
                # Default to 0 or raise error? Let's default to 0 for safety but log warning if needed.
                fast_out = self.fast_adapters[0](x)
            else:
                # Ensure task_id is within bounds
                idx = task_id % self.nb_tasks
                fast_out = self.fast_adapters[idx](x)
        else:
            # Inference: slow + sum(fast_k) (or average)
            # v0 design: sum all fast adapters
            fast_out = 0
            for adapter in self.fast_adapters:
                fast_out = fast_out + adapter(x)
            # Optionally average? The plan said "sum or average". 
            # If we sum, the scale might be large. But let's stick to sum as per "ΔW = slow + Σ fast_k" roughly.
            # Wait, if we sum K adapters, the output magnitude might be K times larger.
            # However, usually only one fast adapter is active during training.
            # If we sum them, we might need to scale. 
            # But let's follow the plan: "ΔW = slow + (平均 or 和) fast_k"
            # Let's use SUM for now, as averaging might dilute the signal if K is large and most are zero.
            # Actually, if most are zero-initialized or small, sum is fine.
            
        # Inference ablation: optionally zero one branch
        mode = getattr(self.config, "nested_lora_eval_ablation", "none")
        if not self.training:
            if mode == "fast_only":
                slow_out = torch.zeros_like(slow_out)
            elif mode == "slow_only":
                fast_out = torch.zeros_like(fast_out)

        return slow_out + fast_out


class AE(nn.Module):
	def __init__(self, config):
		super(AE, self).__init__()
		self.input_dim = config.d_model
		self.config = config
		self.encoder = nn.Linear(self.input_dim, config.rd_dim)
		self.decoder = nn.Linear(config.rd_dim, self.input_dim)
		self.weight_initialize()

	def forward(self, x):
		encoded = self.encoder(x)
		reconstruction = self.decoder(encoded)
		return reconstruction
	
	def compute_reconstruction_loss(self, x):
		x = x.mean(dim=1)
		reconstruction = self.forward(x)
		reconstruction_losses = []
		B = x.shape[0]
		for i in range(B):
			reconstruction_losses.append(self.reconstruction_loss(reconstruction[i], x[i]))
		reconstruction_losses = torch.stack(reconstruction_losses)
		return reconstruction_losses

	def reconstruction_loss(self, reconstruction, x):
		reconstruction_loss = F.mse_loss(reconstruction, x)
		return reconstruction_loss

	def weight_initialize(self):
		with torch.no_grad():
			nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
			nn.init.zeros_(self.encoder.bias)
			nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
			nn.init.zeros_(self.decoder.bias)



class Records:
    def __init__(self, max_len=500) -> None:
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(self._max_len)
        self._mean = 0
        self._var = 0
        self._powersumavg = 0
        self.updating = True

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean
    
    @property
    def stddev(self):
        return math.sqrt(self._var)

    def add_record(self, v):
        if not self.updating:
            return
        if self._curr_len < self._max_len:
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len+len(v)] = v
                self._curr_len += len(v)    
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len   
        else:           
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]
        self._mean = torch.mean(self.record[:self._curr_len])
        self._var = torch.var(self.record[:self._curr_len])
