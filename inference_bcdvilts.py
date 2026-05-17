"""
inference_bcdvilts.py

Unique multimodal inference implementation for BCD-VILTS ( external microservice).
Features:
- Image encoder: ResNet-50 (torchvision), optional Grad-CAM explainability
- Text encoder: BERT (transformers) - uses pooled output and tries to extract attentions
- Genomic encoder: small 1D-CNN -> FC
- Fusion: TransformerEncoder over concatenated modality tokens
- Classifier: MLP head producing logits (binary classification by default)
- Loading: attempts to load a unified checkpoint containing submodule weights
- Predict API: `predict(payload)` => dict with {prediction, confidence, explainability, model_version}
- Designed to be called from a FastAPI server (e.g. inference/api_service.py)

Notes:
- This code is a blueprint; adapt the model checkpoint keys to match how you saved your trained weights.
- Explainability artifacts are lightweight (saliency map + BERT pooled attention snippet).
"""

import os
import json
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

from transformers import AutoTokenizer, AutoModel

_logger = logging.getLogger("bcdvilts_inference")
logging.basicConfig(level=logging.INFO)

# Environment/config defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/bcdvilts_fusion.pt")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))
MAX_GENOMIC_LEN = int(os.environ.get("MAX_GENOMIC_LEN", 1024))
BERT_MODEL_NAME = os.environ.get("BERT_MODEL", "bert-base-uncased")
MODEL_VERSION = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"

# ---------------------------
# Utilities: image/text/genomic
# ---------------------------

_image_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225])
])


def decode_base64_image(b64str: str) -> Optional[Image.Image]:
    try:
        raw = base64.b64decode(b64str)
        img = Image.open(BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        _logger.exception("decode_base64_image failed: %s", e)
        return None


def genomic_to_numeric(seq: str, max_len=MAX_GENOMIC_LEN) -> np.ndarray:
    """
    Convert DNA sequence to numeric encoding (one-hot-lite).
    A/C/G/T -> 0/1/2/3, padded to max_len, returned as float32 array normalized to [0,1].
    """
    if not seq:
        return np.zeros((max_len,), dtype=np.float32)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    arr = np.zeros((max_len,), dtype=np.float32)
    n = min(len(seq), max_len)
    for i in range(n):
        arr[i] = mapping.get(seq[i], 4) / 4.0  # unknowns mapped to 1.0
    return arr


# ---------------------------
# Sub-modules: encoders
# ---------------------------

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Remove classifier head
        modules = list(resnet.children())[:-2]  # keep conv features
        self.feature_extractor = nn.Sequential(*modules)  # outputs [B, 2048, H/32, W/32]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: [B,3,H,W]
        f = self.feature_extractor(x)          # [B,2048,h,w]
        pooled = self.pool(f).view(f.size(0), -1)  # [B,2048]
        out = self.fc(pooled)                  # [B,out_dim]
        return out, f  # return conv features for Grad-CAM if needed


class TextEncoder(nn.Module):
    def __init__(self, model_name=BERT_MODEL_NAME, out_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=False)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)
        self.out_dim = out_dim

    def encode_texts(self, texts: List[str], device=DEVICE):
        # returns tensor [B, out_dim] and optionally attentions (not returned here)
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(device)
        out = self.bert(**enc, return_dict=True)
        pooled = out.pooler_output  # [B, hidden]
        return self.proj(pooled)    # [B, out_dim]


class GenomicEncoder(nn.Module):
    def __init__(self, in_len=MAX_GENOMIC_LEN, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: [B, in_len] float
        x = x.unsqueeze(1)  # [B,1,L]
        x = F.relu(self.conv1(x))  # [B,32,L]
        x = F.relu(self.conv2(x))  # [B,64,L]
        x = self.pool(x).squeeze(-1)  # [B,64]
        out = self.fc(x)  # [B,out_dim]
        return out


# ---------------------------
# Fusion Transformer + Classifier
# ---------------------------

class FusionTransformerModel(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, genomic_dim=256, d_model=512, nhead=8, num_layers=2, num_classes=2):
        super().__init__()
        # Project each modality to d_model
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.genomic_proj = nn.Linear(genomic_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A simple classifier over pooled transformer outputs
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model//2, num_classes)
        )

    def forward(self, image_feat, text_feat, genomic_feat):
        """
        image_feat: [B, image_dim]
        text_feat: [B, text_dim]
        genomic_feat: [B, genomic_dim]
        """
        # Project
        img_tok = self.image_proj(image_feat).unsqueeze(1)     # [B,1,d_model]
        txt_tok = self.text_proj(text_feat).unsqueeze(1)       # [B,1,d_model]
        gen_tok = self.genomic_proj(genomic_feat).unsqueeze(1) # [B,1,d_model]

        # concat tokens -> [B, 3, d_model] then transpose for transformer [S, B, E]
        tokens = torch.cat([img_tok, txt_tok, gen_tok], dim=1)  # [B,3,d_model]
        tokens_t = tokens.transpose(0, 1)  # [3, B, d_model]

        out = self.transformer(tokens_t)   # [3, B, d_model]
        pooled = out.mean(dim=0)           # [B, d_model]
        logits = self.classifier(pooled)   # [B, num_classes]
        return logits


# ---------------------------
# High-level wrapper and model loading
# ---------------------------

class BCDVILTSInferenceModel:
    def __init__(self, device=DEVICE):
        self.device = device
        # instantiate modules
        self.image_encoder = ImageEncoder(out_dim=512, pretrained=True).to(device)
        self.text_encoder = TextEncoder(model_name=BERT_MODEL_NAME, out_dim=512)  # tokenizer + model kept on CPU until to() called
        self.text_encoder.bert.to(device)
        self.text_encoder.proj.to(device)
        self.genomic_encoder = GenomicEncoder(in_len=MAX_GENOMIC_LEN, out_dim=256).to(device)
        self.fusion = FusionTransformerModel(image_dim=512, text_dim=512, genomic_dim=256, d_model=512, nhead=8, num_layers=2, num_classes=2).to(device)

        self.model_version = "init"
        # try loading checkpoint
        if os.path.exists(MODEL_PATH):
            self.load_checkpoint(MODEL_PATH)

    def load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device)
            # Expect keys like: 'image_encoder', 'text_encoder', 'genomic_encoder', 'fusion', 'model_version'
            # Adjust these keys to how you saved your model.
            if isinstance(ckpt, dict):
                if 'image_encoder' in ckpt:
                    self.image_encoder.load_state_dict(ckpt['image_encoder'], strict=False)
                if 'text_encoder' in ckpt:
                    # text encoder likely saved differently; try loading projection & bert if present
                    if 'text_encoder_proj' in ckpt:
                        self.text_encoder.proj.load_state_dict(ckpt['text_encoder_proj'])
                    if 'text_encoder_bert' in ckpt:
                        self.text_encoder.bert.load_state_dict(ckpt['text_encoder_bert'])
                if 'genomic_encoder' in ckpt:
                    self.genomic_encoder.load_state_dict(ckpt['genomic_encoder'])
                if 'fusion' in ckpt:
                    self.fusion.load_state_dict(ckpt['fusion'])
                self.model_version = ckpt.get('model_version', os.path.basename(path))
                _logger.info("Checkpoint loaded. Model version: %s", self.model_version)
            else:
                # If checkpoint is a single state_dict for entire wrapper, try to load into fusion directly
                self.fusion.load_state_dict(ckpt, strict=False)
                self.model_version = os.path.basename(path)
                _logger.info("Generic checkpoint loaded into fusion; model_version set to filename.")
        except Exception as e:
            _logger.exception("Failed to load checkpoint %s : %s", path, e)

    def predict_batch(self, images: List[Image.Image], texts: List[str], genomics: List[str]) -> List[Dict[str, Any]]:
        """
        images: list of PIL images (or empty list per sample)
        texts: list of strings
        genomics: list of strings
        returns: list of dicts {prediction, confidence, explainability: {...}}
        """
        self.image_encoder.eval()
        self.text_encoder.bert.eval()
        self.text_encoder.proj.eval()
        self.genomic_encoder.eval()
        self.fusion.eval()

        batch_size = max(1, max(len(images), len(texts), len(genomics)))
        # prepare tensors
        # images: transform each if present else zeros
        image_tensors = []
        for i in range(batch_size):
            if i < len(images) and images[i] is not None:
                image_tensors.append(_image_transform(images[i]))
            else:
                image_tensors.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        image_batch = torch.stack(image_tensors, dim=0).to(self.device)  # [B,3,H,W]

        # text feats
        text_inputs = [texts[i] if i < len(texts) else "" for i in range(batch_size)]
        text_feats = self.text_encoder.encode_texts(text_inputs, device=self.device)  # [B, text_dim]

        # genomic
        genomic_arrays = []
        for i in range(batch_size):
            seq = genomics[i] if i < len(genomics) else ""
            genomic_arrays.append(torch.tensor(genomic_to_numeric(seq), dtype=torch.float32))
        genomic_batch = torch.stack(genomic_arrays, dim=0).to(self.device)

        # image forward (get conv features too)
        with torch.no_grad():
            image_feats, conv_feats = self.image_encoder(image_batch)  # image_feats [B,512], conv_feats [B,2048,h,w]
            genomic_feats = self.genomic_encoder(genomic_batch)        # [B,256]
            logits = self.fusion(image_feats, text_feats, genomic_feats)  # [B,num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [B,2]

        results = []
        for i in range(batch_size):
            prob = probs[i]
            pred_idx = int(np.argmax(prob))
            confidence = float(prob[pred_idx])
            label = str(pred_idx)  # map to label names later if desired

            # create explainability: for the first image, compute a simple gradient-based saliency if GPU available
            explain = {}
            try:
                # Simple Grad-CAM-like using gradients of classifier w.r.t conv_feats
                # We only compute for first sample to save time; adjust as needed.
                if i == 0 and torch.cuda.is_available():
                    # small grad-cam
                    conv = conv_feats[i:i+1]            # [1,2048,h,w]
                    conv.requires_grad = True
                    # forward path through remaining layers: pool->fc->fusion->classifier trick is complicated;
                    # instead compute logits from image alone by routing through fusion with zeros for other modalities
                    dummy_text = torch.zeros(1, text_feats.size(-1), device=self.device)
                    dummy_gen = torch.zeros(1, genomic_feats.size(-1), device=self.device)
                    img_feat, _ = self.image_encoder(image_batch[i:i+1])
                    logits_img = self.fusion(img_feat, dummy_text, dummy_gen)
                    score = logits_img[0, pred_idx]
                    self.fusion.zero_grad()
                    self.image_encoder.zero_grad()
                    score.backward(retain_graph=True)
                    # conv.grad may be None if not hooked; fall back to simple saliency from image
                    if conv.grad is not None:
                        grad = conv.grad[0].cpu().numpy()  # [C,h,w]
                        weights = np.mean(grad, axis=(1,2))  # [C]
                        cam = np.zeros(conv.shape[2:], dtype=np.float32)
                        for c, w in enumerate(weights):
                            cam += w * conv[0, c].cpu().numpy()
                        cam = np.maximum(cam, 0)
                        cam = cam - cam.min()
                        if cam.max() > 0:
                            cam = cam / cam.max()
                        # resize cam to image size
                        import cv2
                        cam_img = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
                        # overlay
                        pil_img = images[i].resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGBA")
                        heatmap = (255 * cam_img).astype("uint8")
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        heat_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA))
                        overlay = Image.blend(pil_img, heat_pil, alpha=0.4)
                        buff = BytesIO()
                        overlay.save(buff, format="PNG")
                        explain['resnet_heatmap'] = base64.b64encode(buff.getvalue()).decode("utf-8")
                    else:
                        explain['resnet_heatmap'] = None
                else:
                    explain['resnet_heatmap'] = None
            except Exception as e:
                _logger.exception("Explainability generation failed: %s", e)
                explain['resnet_heatmap'] = None

            # BERT attention: we didn't set output_attentions True; include placeholder or token count
            explain['bert_tokens'] = len(text_inputs[i].split()) if text_inputs[i] else 0

            results.append({
                "prediction": label,
                "confidence": confidence,
                "explainability": explain,
                "model_version": self.model_version
            })

        return results


# ---------------------------
# Single-call predict function (for API)
# ---------------------------

# Create a module-level model instance to reuse across calls (loaded at import)
_MODEL_INSTANCE: Optional[BCDVILTSInferenceModel] = None


def _get_model_instance():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = BCDVILTSInferenceModel(device=DEVICE)
    return _MODEL_INSTANCE


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload expected shape:
    {
      "images": [ "<base64-encoded-image>", ... ]  OR "image": "<base64>",
      "text": "clinical text...",
      "genomic": "ATCG..."
    }

    Returns:
    {
      "prediction": "0" or "1",
      "confidence": float,
      "explainability": {...},
      "model_version": "..."
    }
    """
    try:
        images_b64 = payload.get("images") or ( [payload["image"]] if payload.get("image") else [] )
        images = [decode_base64_image(b) for b in images_b64] if images_b64 else []

        # support batch (if lists) or single
        if isinstance(payload.get("text"), list):
            texts = payload.get("text")
        else:
            texts = [payload.get("text", "")] if images or payload.get("text") else [""]

        if isinstance(payload.get("genomic"), list):
            genomics = payload.get("genomic")
        else:
            genomics = [payload.get("genomic", "")]

        # align lengths: choose batch_size = max(len(images), len(texts), len(genomics))
        batch_size = max(1, len(images), len(texts), len(genomics))
        # pad lists
        while len(images) < batch_size:
            images.append(None)
        while len(texts) < batch_size:
            texts.append("")
        while len(genomics) < batch_size:
            genomics.append("")

        model = _get_model_instance()
        results = model.predict_batch(images, texts, genomics)
        # if batch_size==1 return first result for convenience
        if len(results) == 1:
            return results[0]
        else:
            return {"batch_results": results, "model_version": model.model_version}
    except Exception as e:
        _logger.exception("Predict failed: %s", e)
        return {"error": str(e), "model_version": getattr(_MODEL_INSTANCE, "model_version", "none")}


# ---------------------------
# CLI Runner (for local testing)
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run BCD-VILTS multimodal inference (unique implementation)")
    parser.add_argument("--input", "-i", required=True, help="Path to JSON input file")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help="Optional model checkpoint to load")
    parser.add_argument("--out", "-o", help="Optional output file path")
    args = parser.parse_args()

    if args.model and os.path.exists(args.model):
        MODEL_PATH = args.model  # override global
    _logger.info("Device: %s", DEVICE)
    with open(args.input, "r") as f:
        payload = json.load(f)

    out = predict(payload)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fo:
            json.dump(out, fo, indent=2)
