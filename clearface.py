import paddlehub as hub

# 加载模型
# 可根据需要更换上述模型中的任何一个
module_name = 'falsr_c'
sr_model = hub.Module(module_name)

# 设置图像路径
# 可同时预测多张图片
paths = ['./avatar/out.png']

# 调用预测接口
# 具体参数设置请参考前文所述
res = sr_model.reconstruct(
    images=None,
    paths=paths,
    use_gpu=True,
    visualization=True,
	output_dir='%s_output' % module_name
)
