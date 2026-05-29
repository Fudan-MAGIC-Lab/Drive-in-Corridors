import torch
from torch.autograd import gradcheck

# 假设QPFunctionFn的代码位于名为qp_solver的模块中
from qp import QPFunction

def test_qp_gradient():
    # 设置求解器为PDIPM_BATCHED（确保代码中有对应的solver选项）
    # 注意：根据实际代码调整solver的导入和设置方式
    global solver
    solver = QPFunction.QPFunctionFn.QPSolvers.PDIPM_BATCHED  # 假设有此枚举类

    # 测试参数（单样本，无批次维度）
    nz = 2  # 变量数
    nineq = 1  # 不等式约束数
    neq = 0  # 等式约束数

    # 构造一个简单的QP问题，已知最优解和梯度
    Q = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)  # 正定矩阵
    p = torch.tensor([-1.0, -1.0], requires_grad=True)
    G = torch.tensor([[1.0, 1.0]], requires_grad=True)
    h = torch.tensor([1.0], requires_grad=True)
    A = torch.empty((0, nz))  # 无等式约束
    b = torch.empty((0,))

    # 前向计算
    def forward_func(Q, p, G, h, A, b):
        # 包装成参数，注意处理无等式约束的情况
        # 注意：根据QPFunctionFn的要求，可能需要将空张量设为特定形式
        A = torch.empty(0, nz) if neq == 0 else A
        b = torch.empty(0) if neq == 0 else b
        return QPFunctionFn.apply(Q, p, G, h, A, b)

    # 使用gradcheck进行梯度验证（需转换为double类型）
    inputs = (Q.double().requires_grad_(True),
              p.double().requires_grad_(True),
              G.double().requires_grad_(True),
              h.double().requires_grad_(True),
              A.double().requires_grad_(False),  # 无梯度
              b.double().requires_grad_(False))

    # 执行梯度检查，设置eps和atol以适应数值误差
    test = gradcheck(forward_func, inputs, eps=1e-6, atol=1e-4, check_undefined_grad=False)
    print("Gradcheck for single sample (no batch):", test)

    # 测试带批次的情况（批次大小为2）
    nBatch = 2
    Q_batch = Q.unsqueeze(0).repeat(nBatch, 1, 1).double().requires_grad_(True)
    p_batch = p.unsqueeze(0).repeat(nBatch, 1).double().requires_grad_(True)
    G_batch = G.unsqueeze(0).repeat(nBatch, 1, 1).double().requires_grad_(True)
    h_batch = h.unsqueeze(0).repeat(nBatch, 1).double().requires_grad_(True)
    A_batch = A.unsqueeze(0).repeat(nBatch, 1, 1).double().requires_grad_(False)
    b_batch = b.unsqueeze(0).repeat(nBatch, 1).double().requires_grad_(False)

    inputs_batch = (Q_batch, p_batch, G_batch, h_batch, A_batch, b_batch)
    test_batch = gradcheck(forward_func, inputs_batch, eps=1e-6, atol=1e-4, check_undefined_grad=False)
    print("Gradcheck for batch samples:", test_batch)

    # 测试部分参数共享的情况（例如Q在所有批次中相同）
    Q_shared = Q.unsqueeze(0).expand(nBatch, nz, nz).double().requires_grad_(True)
    # 其他参数不同
    p_batch = torch.randn(nBatch, nz).double().requires_grad_(True)
    G_batch = torch.randn(nBatch, nineq, nz).double().requires_grad_(True)
    h_batch = torch.randn(nBatch, nineq).double().requires_grad_(True)

    inputs_shared = (Q_shared, p_batch, G_batch, h_batch, A_batch, b_batch)
    test_shared = gradcheck(forward_func, inputs_shared, eps=1e-6, atol=1e-4, check_undefined_grad=False)
    print("Gradcheck with shared Q:", test_shared)

if __name__ == "__main__":
    test_qp_gradient()