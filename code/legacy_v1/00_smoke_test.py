"""阶段0验收烟雾测试：最小分割模型的前向 / 反向 / 保存 / 恢复。

用法：
    .venv/bin/python 02_scripts/modeling/00_smoke_test.py

验收门槛：
    - 两张 5090 均被 PyTorch 独立识别（device_count == 2）。
    - 一个小型分割模型能在每张 GPU 上完成前向、反向、保存和恢复。
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def smoke(device: str) -> None:
    # 8 通道输入对应"四季 LST(4) + 光学波段(4)"，1 通道输出为隐伏断裂候选概率
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=8, classes=1).to(device)
    x = torch.randn(2, 8, 128, 128, device=device)
    y = torch.rand(2, 1, 128, 128, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()

    # 前向 + 反向 + 参数更新
    out = model(x)
    loss = crit(out, y)
    loss.backward()
    opt.step()

    # 保存（保存更新后的权重）
    torch.save(model.state_dict(), "/tmp/smoke.pt")

    # 恢复
    restored = smp.Unet("resnet18", encoder_weights=None, in_channels=8, classes=1).to(device)
    restored.load_state_dict(torch.load("/tmp/smoke.pt", weights_only=True))

    # 校验1：恢复后参数与原模型完全一致（保存/恢复是否无损）
    for (k, v), (k2, v2) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert k == k2, f"参数名不一致: {k} vs {k2}"
        assert torch.allclose(v, v2, atol=1e-7), f"参数 {k} 恢复不一致"

    # 校验2：eval 模式下同一输入前向输出一致（避免 BN 训练态随机性）
    model.eval()
    restored.eval()
    with torch.no_grad():
        o1 = model(x)
        o2 = restored(x)
    assert torch.allclose(o1, o2, atol=1e-5), "前向输出不一致"

    print(f"[OK] {device}: fwd/bwd/save/restore 通过, loss={loss.item():.4f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA 不可用"
    n = torch.cuda.device_count()
    print(f"CUDA 可用, device_count = {n}")
    assert n == 2, f"期望 2 张 GPU，实际 {n}"
    for i in range(n):
        smoke(f"cuda:{i}")
    print("SMOKE TEST PASSED")
