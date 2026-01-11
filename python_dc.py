import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
points = np.array([
    [0, 0],  # A
    [1, 0],  # B
    [3, 0],  # C
    [6, 0],  # D
])
labels = ['A', 'B', 'C', 'D']
N = len(points)
print("\n================ BƯỚC 1: TÍNH MA TRẬN KHOẢNG CÁCH =================")
dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist[i, j] = np.linalg.norm(points[i] - points[j])
print("Ma trận khoảng cách Euclidean:")
print(np.round(dist, 2))
print("\n================ BƯỚC 2: CHỌN NGƯỠNG dc =================")
dc = 2.5
print(f"Giá trị dc được chọn thủ công: dc = {dc}")
print("\n================ BƯỚC 3: TÍNH MẬT ĐỘ CỤC BỘ ρ =================")
print("Công thức DPC gốc (hàm bước): ρᵢ = Σ χ(dᵢⱼ − dc)")
rho = np.zeros(N)
for i in range(N):
    rho[i] = np.sum(dist[i] < dc) - 1   # trừ chính nó
print("Mật độ cục bộ ρ:")
for i in range(N):
    print(f"ρ({labels[i]}) = {int(rho[i])}")
print("\n================ BƯỚC 4: TÍNH KHOẢNG CÁCH TÁCH BIỆT δ =================")
print("δᵢ = khoảng cách nhỏ nhất tới điểm có mật độ cao hơn")
print("Với điểm có mật độ cao nhất: δᵢ = max(dᵢⱼ)")
delta = np.zeros(N)
rho_max = np.max(rho)
for i in range(N):
    higher = np.where(rho > rho[i])[0]
    if rho[i] == rho_max:
        delta[i] = np.max(dist[i])
    else:
        delta[i] = np.min(dist[i, higher])
for i in range(N):
    print(f"δ({labels[i]}) = {delta[i]:g}")
print("\n================ BƯỚC 5: TÍNH GIÁ TRỊ QUYẾT ĐỊNH γ =================")
print("Công thức: γᵢ = ρᵢ × δᵢ")
gamma = rho * delta
for i in range(N):
    print(f"γ({labels[i]}) = {gamma[i]:g}")
print("\n================ BƯỚC 6: XÁC ĐỊNH TÂM CỤM =================")
center = np.argmax(gamma)
print(f"Tâm cụm được xác định là: {labels[center]} (γ lớn nhất)")
print("\n================ BƯỚC 7: VẼ DECISION GRAPH (ρ–δ) =================")
plt.figure(figsize=(6, 4))
plt.scatter(rho, delta, s=80)
for i in range(N):
    plt.text(rho[i] + 0.02, delta[i] + 0.02, labels[i])
plt.xlabel("Mật độ cục bộ ρ")
plt.ylabel("Khoảng cách tách biệt δ")
plt.title("Decision Graph của thuật toán DPC (ρ–δ)")
plt.grid(True)
plt.show()  
print("\n================ BƯỚC 8: VẼ DỮ LIỆU & TÂM CỤM (MINH HỌA) =================")
plt.figure(figsize=(7, 4))
for i in range(N):
    plt.scatter(points[i, 0], points[i, 1],
                s=80,
                color='blue')
    plt.text(points[i, 0] + 0.05,
             points[i, 1] + 0.05,
             labels[i])
# Đánh dấu tâm cụm đã chọn (từ decision graph)
plt.scatter(points[center, 0],
            points[center, 1],
            s=200,
            color='yellow',
            edgecolors='black',
            marker='X',
            label='Tâm cụm tiềm năng')
plt.title("Dữ liệu ban đầu và tâm cụm tiềm năng (DPC gốc)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
