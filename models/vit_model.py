# models/vit_model.py  (compatible with old/new torchvision)
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# torchvision ViT ctors + weights
from torchvision.models.vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights

_NAME2CTOR = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights),
    # aliases (timm-like) -> torchvision
    "vit_base_patch16_224": (vit_b_16, ViT_B_16_Weights),
    "vit_base_patch32_224": (vit_b_32, ViT_B_32_Weights),
    "vit_large_patch16_224": (vit_l_16, ViT_L_16_Weights),
    "vit_large_patch32_224": (vit_l_32, ViT_L_32_Weights),
}

class ViTModel(nn.Module):
    """
    - ใช้ ViT ของ torchvision (ไม่พึ่ง timm)
    - รวม classifier เข้าโมเดล (เทรน/val ใช้เส้นทางเดียวกัน)
    - extract_embeddings(): ใช้ตอนคำนวณ EER (L2-norm เมื่อขอ)
    - pool: 'mean' (แนะนำ) หรือ 'cls'
    """
    def __init__(
        self,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        num_classes: int = 0,
        pool: Literal["mean", "cls"] = "mean",
        drop_path: float = 0.0,   # ถ้า torchvision เก่า จะเพิกเฉยพารามฯนี้
    ):
        super().__init__()
        assert pool in ("mean", "cls")
        ctor, weight_cls = _NAME2CTOR.get(model_name, (vit_b_16, ViT_B_16_Weights))
        weights = weight_cls.IMAGENET1K_V1 if pretrained else None

        # create backbone (handle old/new torchvision)
        try:
            self.backbone = ctor(weights=weights, dropout=0.0, stochastic_depth_prob=drop_path)
        except TypeError:
            if drop_path and drop_path > 0:
                warnings.warn("torchvision ViT: ไม่รองรับ stochastic_depth_prob; ใช้ค่าเริ่มต้น (ไม่มี drop_path)")
            self.backbone = ctor(weights=weights, dropout=0.0)

        # remove original head & get embed dim
        self.embed_dim = None
        if hasattr(self.backbone, "heads"):
            if hasattr(self.backbone.heads, "head") and isinstance(self.backbone.heads.head, nn.Linear):
                self.embed_dim = self.backbone.heads.head.in_features
                self.backbone.heads.head = nn.Identity()
            else:
                last_linear = None
                for m in reversed(list(self.backbone.heads.modules())):
                    if isinstance(m, nn.Linear):
                        last_linear = m; break
                if last_linear is not None:
                    self.embed_dim = last_linear.in_features
                self.backbone.heads = nn.Identity()
        if self.embed_dim is None and hasattr(self.backbone, "hidden_dim"):
            self.embed_dim = self.backbone.hidden_dim
        if self.embed_dim is None:
            raise RuntimeError("ไม่พบขนาด embedding ของ ViT (embed_dim)")

        self.pool = pool
        self.classifier = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else None

    # ----- feature path -----
    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1+N, D): index 0 = CLS, 1..N = patch tokens
        x = self.backbone._process_input(x)                     # (B, N, D)
        n = x.shape[0]
        cls_token = self.backbone.class_token.expand(n, -1, -1) # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)                    # (B, 1+N, D)
        x = self.backbone.encoder(x)                            # (B, 1+N, D)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        if self.pool == "cls":
            feat = tokens[:, 0]              # (B, D)
        else:
            feat = tokens[:, 1:].mean(dim=1) # (B, D)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)      # (B, D)
        if self.classifier is None:
            return feat
        return self.classifier(feat)         # (B, C)

    @torch.no_grad()
    def extract_embeddings(self, x: torch.Tensor, l2norm: bool = True) -> torch.Tensor:
        self.eval()
        feat = self.forward_features(x)
        if l2norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat

    # ----- freeze / unfreeze -----
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 4):
        try:
            layers = list(self.backbone.encoder.layers)  # nn.Sequential ของ encoder blocks
            depth = len(layers)
            keep = max(0, depth - n_blocks)
            for i, layer in enumerate(layers):
                req = (i >= keep)
                for p in layer.parameters():
                    p.requires_grad = req
            if hasattr(self.backbone.encoder, "ln"):
                for p in self.backbone.encoder.ln.parameters():
                    p.requires_grad = True
        except Exception:
            for p in self.backbone.parameters():
                p.requires_grad = True
