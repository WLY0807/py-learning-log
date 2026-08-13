import numpy as np

# ===== 1. 生成模拟数据 =====
np.random.seed(42)
n = 200                          # 样本数
X = np.random.randn(n, 3)        # 3个特征
true_w = np.array([[2.0], [-1.5], [0.8]])  # 真实权重
true_b = 4.0
y = X @ true_w + true_b + 0.3 * np.random.randn(n, 1)

# ===== 2. 初始化参数 =====
w = np.random.randn(3, 1) * 0.01
b = 0.0
lr = 0.1
epochs = 500

# ===== 3. 训练循环（你需要补全）=====
for epoch in range(epochs):
    # 前向传播：计算预测值 y_pred
    y_pred = X @ w + b

    # 计算损失 MSE
    e = y_pred - y
    e1 = e ** 2
    loss = np.mean(e1)

    # 反向传播：计算梯度   loss.backward() 
    tidu_w = (2 / n) * X.T @ e
    tidu_b = 2 * np.mean(e)
    
    
    # 更新参数   updater.step()
    w -= lr * tidu_w
    b -= lr * tidu_b
    
    # 每50轮打印一次损失
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss = {loss.item():.6f}")

# ===== 4. 验证结果 =====
print(f"\n真实的 w: {true_w.ravel()}")
print(f"学到的 w: {w.ravel()}")
print(f"真实的 b: {true_b}")
print(f"学到的 b: {b.item():.4f}")

'''
跑出的结果：
epoch   0 | loss = 25.089735
epoch  50 | loss = 0.091128
epoch 100 | loss = 0.091128
epoch 150 | loss = 0.091128
epoch 200 | loss = 0.091128
epoch 250 | loss = 0.091128
epoch 300 | loss = 0.091128
epoch 350 | loss = 0.091128
epoch 400 | loss = 0.091128
epoch 450 | loss = 0.091128

真实的 w: [ 2.  -1.5  0.8]
学到的 w: [ 1.99284879 -1.4722891   0.83556342]
真实的 b: 4.0
学到的 b: 4.0056
'''