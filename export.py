import torch
from ultralytics import YOLO

print("-> 1. 加载 PyTorch 权重...")
model = YOLO('yolo26n-seg.pt')  # 确保这里是你的权重名
num_classes = len(model.names)
print(f"   检测到模型包含类别数: {num_classes}")

print("-> 2. 禁用端到端 NMS 融合...")
model.model.end2end = False
for m in model.model.modules():
    if hasattr(m, 'end2end'):
        m.end2end = False
    if hasattr(m, 'format'):
        m.format = 'onnx'

print("-> 3. 注入【全自动张量雷达与缝合器】...")
old_forward = model.model.forward


# 💡 核心组件 1：递归张量提取器（无论包得多深，全部挖出来）
def flatten_outputs(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(flatten_outputs(item))
        return res
    return []


def new_split_forward(x, *args, **kwargs):
    out = old_forward(x, *args, **kwargs)

    # 拿到所有的原始张量
    all_tensors = flatten_outputs(out)

    protos = None
    detect_tensors = []

    # 💡 核心组件 2：智能雷达分类
    for t in all_tensors:
        if t.dim() == 4:
            protos = t  # 抓到 4 维的原型图 [1, 32, 160, 160]
        elif t.dim() == 3:
            detect_tensors.append(t)  # 抓到 3 维的检测网格

    if protos is None:
        raise ValueError("严重异常：未能在输出中找到 4 维的掩码原型图！")
    if len(detect_tensors) == 0:
        raise ValueError("严重异常：未能在输出中找到 3 维的检测张量！")

    # 💡 核心组件 3：智能张量缝合
    if len(detect_tensors) == 1:
        main_out = detect_tensors[0]
    else:
        # 场景 A: 按特征层拆分了 (P3, P4, P5)。通道数一样，锚点数不同
        if detect_tensors[0].shape[1] == detect_tensors[-1].shape[1]:
            main_out = torch.cat(detect_tensors, dim=2)  # 在 8400 这个维度上强行拼接
        # 场景 B: 按属性拆分了 (boxes, scores, masks)。锚点数一样，通道数不同
        elif detect_tensors[0].shape[2] == detect_tensors[-1].shape[2]:
            main_out = torch.cat(detect_tensors, dim=1)  # 在 116 这个维度上强行拼接
        else:
            raise RuntimeError(f"无法识别的 3 维张量组合: {[t.shape for t in detect_tensors]}")

    # 🔪 最终物理切割：main_out 现在绝对是 [1, 116, 8400]！
    boxes = main_out[:, :4, :]
    scores = main_out[:, 4: 4 + num_classes, :]
    masks = main_out[:, 4 + num_classes:, :]

    # 完美返回 4 个独立的张量
    return boxes, scores, masks, protos


model.model.forward = new_split_forward

print("-> 4. 开始导出极致物理分离版 ONNX...")
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    half=False,
    dynamic=False
)
print("🎉 手术圆满成功！无敌的 4 输出分离版 ONNX 已经诞生！")