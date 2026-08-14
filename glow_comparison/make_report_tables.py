"""Generate markdown metric tables from results/metrics.json."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))

syn = [k for k in m if k.startswith("syn")]
real = [k for k in m if not k.startswith("syn")]

print("### 合成集（有真值 glow，越高/越低见表头）\n")
print("| 场景 | PSNR↑ ONVE | PSNR↑ Li2015 | SSIM↑ ONVE | SSIM↑ Li2015 | MAE↓ ONVE | MAE↓ Li2015 |")
print("|---|---|---|---|---|---|---|")
for k in syn:
    o, l = m[k]["onve"], m[k]["li2015"]
    def b(a, c, hi=True):
        best = (a > c) if hi else (a < c)
        return (f"**{a:.3f}**" if best else f"{a:.3f}",
                f"**{c:.3f}**" if not best else f"{c:.3f}")
    p = b(o["psnr"], l["psnr"]); s = b(o["ssim"], l["ssim"]); e = b(o["mae"], l["mae"], hi=False)
    print(f"| {k} | {p[0]} | {p[1]} | {s[0]} | {s[1]} | {e[0]} | {e[1]} |")
avg = {}
for side in ["onve", "li2015"]:
    avg[side] = {x: sum(m[k][side][x] for k in syn) / len(syn) for x in ["psnr", "ssim", "mae"]}
print(f"| **平均** | {avg['onve']['psnr']:.2f} | {avg['li2015']['psnr']:.2f} | "
      f"{avg['onve']['ssim']:.3f} | {avg['li2015']['ssim']:.3f} | "
      f"{avg['onve']['mae']:.4f} | {avg['li2015']['mae']:.4f} |")

print("\n### 全部图像（无参考指标）\n")
print("| 图像 | 平滑度↓ ONVE | 平滑度↓ Li | 泄漏↓ ONVE | 泄漏↓ Li | 负残差↓ ONVE | 负残差↓ Li | 耗时 ONVE | 耗时 Li |")
print("|---|---|---|---|---|---|---|---|---|")
for k in syn + real:
    o, l = m[k]["onve"], m[k]["li2015"]
    print(f"| {k} | {o['smoothness']:.4f} | {l['smoothness']:.4f} | "
          f"{o['leakage']:.3f} | {l['leakage']:.3f} | "
          f"{o['neg_residual']:.4f} | {l['neg_residual']:.4f} | "
          f"{m[k]['onve_seconds']:.0f}s | {m[k]['li_seconds']:.2f}s |")
