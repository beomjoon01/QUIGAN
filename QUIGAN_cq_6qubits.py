from pathlib import Path
import math

import torch
import numpy as np
import pandas as pd
from scipy.stats import unitary_group

if __package__:
    from .exact_qui_cq import exact_qui
    from .mqui_cq import measured_qui
else:
    from exact_qui_cq import exact_qui
    from mqui_cq import measured_qui

# ==========================================
# Hyperparameters & Target setting
# Target CQ channel: 6 qubits (X:B = 3:3); environment and ancillas are separate.
# Circuit depths and ancilla sizes are tunable hyperparameters.
# ==========================================
layers_G = 8
layers_D = 256
master_seed = 0  # Set an integer to reproduce the channel; None draws a new channel.
# used master seed = 1000000, 1000001, 1000002, 1000004, 1000007, 1000008, 1000011, 1000013, 1000016, 1000017, 1000020, 1000021, 1000023, 1000024, 1000025, 1000029, 1000031, 1000035, 1000038, 1000040, 1000041, 1000043, 1000045, 1000049, 1000054, 1000055, 1000060, 1000062, 1000063, 1000064  # 30 accepted production targets, sorted; the active seed=0 above remains a usage example.

N_X = 3
N_B = 3
N_E = N_B  # Haar reduction gives full-rank outputs with probability one.
N_anc_D = 0
N_anc_G = 3

num_inputs = 2 ** N_X
num_qubits = N_X + N_B + N_anc_D

wires_X = list(range(0, N_X))
wires_B = list(range(N_X, N_X + N_B))
wires_E = list(range(N_X + N_B, N_X + N_B + N_E))
wires_anc_D = list(range(N_X + N_B + N_E, N_X + N_B + N_E + N_anc_D))
wires_anc_G = list(range(N_X + N_B + N_E + N_anc_D, N_X + N_B + N_E + N_anc_D + N_anc_G))

total_wires = N_X + N_B + N_E + N_anc_D + N_anc_G

# Target CQ channel: one preparation unitary on BE per input x.
master_rng = np.random.Generator(np.random.PCG64(master_seed))
target_seeds = master_rng.integers(0, 2 ** 32, size=num_inputs, dtype=np.uint32)
U_targets_np = np.stack([
    unitary_group.rvs(2 ** (N_B + N_E), random_state=int(seed))
    for seed in target_seeds
])
U_targets = torch.tensor(U_targets_np, dtype=torch.complex128)

# rho_x = Tr_E[U_x |0><0| U_x^dagger], computed once for the fixed channel.
target_vectors = U_targets_np[:, :, 0].reshape(num_inputs, 2 ** N_B, 2 ** N_E)
rho_targets = target_vectors @ target_vectors.conj().transpose(0, 2, 1)
if not np.all(np.linalg.eigvalsh(rho_targets) > 0):
    raise ValueError("Target outputs must be full rank; choose a different master_seed.")

# ==========================================
# Pure PyTorch Quantum Simulator
# ==========================================
class DifferentiableQuantumSimulator:
    def __init__(self, num_qubits, device='cpu'):
        self.num_qubits = num_qubits
        self.device = device
        # 초기 상태 |0...0>
        self.state = torch.zeros((2,) * num_qubits, dtype=torch.complex128, device=device)
        self.state[(0,) * num_qubits] = 1.0 + 0.0j

    def apply_gate(self, matrix, wires):
        k = len(wires)
        matrix = matrix.to(self.device).reshape((2,) * (2 * k))
        
        # 텐서 내적(Tensordot)을 이용한 게이트 연산
        state = torch.tensordot(matrix, self.state, dims=(list(range(k, 2 * k)), wires))
        
        # 연산 후 뒤섞인 텐서 차원을 원래 큐비트 순서대로 복원
        rem_wires = [w for w in range(self.num_qubits) if w not in wires]
        perm = [0] * self.num_qubits
        for i, w in enumerate(wires):
            perm[w] = i
        for i, w in enumerate(rem_wires):
            perm[w] = i + k
            
        self.state = state.permute(tuple(perm))

    def probs(self, wires):
        # 지정되지 않은 와이어(환경) 추적(Trace out)
        prob_tensor = torch.abs(self.state) ** 2
        rem_wires = tuple(w for w in range(self.num_qubits) if w not in wires)
        if rem_wires:
            prob_tensor = torch.sum(prob_tensor, dim=rem_wires)
            
        # 큐비트 순서 정렬
        sorted_wires = sorted(wires)
        perm = [sorted_wires.index(w) for w in wires]
        prob_tensor = prob_tensor.permute(tuple(perm))
        
        return prob_tensor.reshape(-1)

    def get_density_matrix(self, wires):
        # 지정된 와이어의 밀도 행렬(Density Matrix) 계산
        rem_wires = tuple(w for w in range(self.num_qubits) if w not in wires)
        perm = list(wires) + list(rem_wires)
        state_perm = self.state.permute(tuple(perm))
        
        dim_wires = 2 ** len(wires)
        dim_rem = 2 ** len(rem_wires)
        state_mat = state_perm.reshape(dim_wires, dim_rem)
        
        rho = state_mat @ state_mat.conj().T
        return rho.detach().cpu().numpy()

# ------------------------------------------
# 양자 게이트 정의 (미분 가능하도록 torch로 구현)
# ------------------------------------------
def RX(theta):
    cos = torch.cos(theta / 2).to(torch.complex128)
    sin = torch.sin(theta / 2).to(torch.complex128)
    return torch.stack([
        torch.stack([cos, -1j * sin]),
        torch.stack([-1j * sin, cos])
    ])

def RY(theta):
    cos = torch.cos(theta / 2).to(torch.complex128)
    sin = torch.sin(theta / 2).to(torch.complex128)
    return torch.stack([
        torch.stack([cos, -sin]),
        torch.stack([sin, cos])
    ])

def RZ(theta):
    exp_plus = torch.exp(-1j * theta / 2)
    exp_minus = torch.exp(1j * theta / 2)
    zero = torch.zeros_like(theta, dtype=torch.complex128)
    return torch.stack([
        torch.stack([exp_plus, zero]),
        torch.stack([zero, exp_minus])
    ])

CNOT = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=torch.complex128)

# ==========================================
# Ansatz structure & PQC
# ==========================================
def hardware_efficient_ansatz(sim, params, wires):
    layers = params.shape[0]
    for l in range(layers):
        for i, wire in enumerate(wires):
            sim.apply_gate(RX(params[l, i, 0]), [wire])
            sim.apply_gate(RY(params[l, i, 1]), [wire])
            sim.apply_gate(RZ(params[l, i, 2]), [wire])
        if len(wires) > 1:
            for i in range(len(wires)):
                sim.apply_gate(CNOT, [wires[i], wires[(i + 1) % len(wires)]])

def prepare_input(x):
    sim = DifferentiableQuantumSimulator(total_wires)
    sim.state.zero_()
    basis = [0] * total_wires
    for i, wire in enumerate(wires_X):
        basis[wire] = (x >> (N_X - 1 - i)) & 1
    sim.state[tuple(basis)] = 1.0 + 0.0j
    return sim

# [Part 1] P(i) = sum_x p_x P(i|x), with a common generator state sigma_B.
def circuit_p(theta, phi, input_probs):
    conditional_probs = []
    for x in range(num_inputs):
        sim = prepare_input(x)
        hardware_efficient_ansatz(sim, theta, wires=wires_B + wires_anc_G)
        hardware_efficient_ansatz(sim, phi, wires=wires_X + wires_B + wires_anc_D)
        conditional_probs.append(sim.probs(wires_X + wires_B + wires_anc_D))
    return input_probs @ torch.stack(conditional_probs)

# [Part 2] Q(i) = sum_x p_x Q(i|x), using the fixed channel output rho_x.
def circuit_q(phi, input_probs):
    conditional_probs = []
    for x in range(num_inputs):
        sim = prepare_input(x)
        sim.apply_gate(U_targets[x], wires_B + wires_E)
        hardware_efficient_ansatz(sim, phi, wires=wires_X + wires_B + wires_anc_D)
        conditional_probs.append(sim.probs(wires_X + wires_B + wires_anc_D))
    return input_probs @ torch.stack(conditional_probs)

# ==========================================
# Channel reference calculation (NumPy)
# ==========================================
# 고정된 채널의 reference를 각각 한 번 계산한다 (nats).
exact_qui_result = exact_qui(rho_targets, tolerance=1e-6, max_retries=3)
if not exact_qui_result["accepted"]:
    raise RuntimeError(f"QUI reference gap {exact_qui_result['gap']:.3e} exceeds tolerance.")
exact_val = exact_qui_result["midpoint"]
print(f"*** Exact QUI reference: {exact_val:.10f} "
      f"[{exact_qui_result['lower']:.10f}, {exact_qui_result['upper']:.10f}] ***")

exact_mqui = measured_qui(rho_targets, tolerance=1e-6, max_retries=3)
if not exact_mqui["accepted"]:
    raise RuntimeError(f"mQUI reference gap {exact_mqui['gap']:.3e} exceeds tolerance.")
print(f"*** Measured QUI reference: {exact_mqui['midpoint']:.10f} "
      f"[{exact_mqui['lower']:.10f}, {exact_mqui['upper']:.10f}] ***")

# ==========================================
# Loss function
# ==========================================

def compute_loss(theta, phi, w_params, input_probs):
    # 평균을 빼서 f_w_vals를 중심화한다.
    f_w_vals = w_params - torch.mean(w_params)
    
    p_probs = circuit_p(theta, phi, input_probs)
    q_probs = circuit_q(phi, input_probs)
    
    # ------------------------------------------------
    # Term 1: E_P[f(i)]
    # ------------------------------------------------
    term1 = torch.sum(f_w_vals * p_probs)
    
    # ------------------------------------------------
    # Term 2: epsilon-free log(E_Q[e^f]); keep derivatives at Q_i == 0.
    # ------------------------------------------------
    shift = f_w_vals.max().detach()
    term2 = torch.log(torch.sum(q_probs * torch.exp(f_w_vals - shift)))
    
    return term1 - shift - term2

# ==========================================
# Training
# ==========================================
epochs = 4000
num_runs = 10


def project_simplex(values):
    # Euclidean projection onto p_x >= 0 and sum_x p_x = 1.
    sorted_values = torch.sort(values, descending=True).values
    cumulative = torch.cumsum(sorted_values, dim=0) - 1
    counts = torch.arange(1, values.numel() + 1, dtype=values.dtype, device=values.device)
    active = torch.nonzero(sorted_values > cumulative / counts, as_tuple=True)[0][-1]
    threshold = cumulative[active] / counts[active]
    return torch.clamp(values - threshold, min=0)


def train_once(run_index):
    # Fresh parameters and optimizer states for each independent training run.
    theta = torch.randn(layers_G, N_B + N_anc_G, 3, dtype=torch.float64, requires_grad=True)
    phi = (torch.randn(layers_D, num_qubits, 3, dtype=torch.float64) * 0.1).requires_grad_()
    w_params = torch.zeros(2**num_qubits, dtype=torch.float64, requires_grad=True)
    input_probs = torch.full((num_inputs,), 1.0 / num_inputs, dtype=torch.float64, requires_grad=True)

    opt_G = torch.optim.Adam([theta], lr=0.001)
    opt_phi = torch.optim.Adam([phi], lr=0.001)
    opt_w = torch.optim.Adam([w_params], lr=0.05)
    lr_p = 0.001

    loss_array = []
    p_array = []
    # One-based schedule: first half flat, second half cosine to 0.1 at T.
    for epoch in range(epochs):
        progress = max(0.0, ((epoch + 1) - epochs / 2) / (epochs / 2))
        lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for optimizer, initial_lr in ((opt_G, 0.001), (opt_phi, 0.001), (opt_w, 0.05)):
            for group in optimizer.param_groups:
                group['lr'] = initial_lr * lr_scale
        # ------------------------------------------------
        # Step 1: Discriminator
        # ------------------------------------------------
        for _ in range(10):
            opt_phi.zero_grad()
            opt_w.zero_grad()

            loss_D = -1 * compute_loss(theta.detach(), phi, w_params, input_probs.detach())
            loss_D.backward()

            opt_phi.step()
            opt_w.step()

        # ------------------------------------------------
        # Step 2: Generator
        # ------------------------------------------------
        for _ in range(1):
            opt_G.zero_grad()
            loss_G = compute_loss(theta, phi.detach(), w_params.detach(), input_probs.detach())
            loss_G.backward()
            opt_G.step()

        # ------------------------------------------------
        # Step 3: Input distribution (projected gradient ascent)
        # ------------------------------------------------
        for _ in range(1):
            loss_p = compute_loss(theta.detach(), phi.detach(), w_params.detach(), input_probs)
            grad_p, = torch.autograd.grad(loss_p, input_probs)
            with torch.no_grad():
                input_probs.copy_(project_simplex(input_probs + lr_p * lr_scale * grad_p))

        # Record the current objective and input distribution after all D/G/p updates.
        with torch.no_grad():
            current_est = compute_loss(theta, phi, w_params, input_probs).item()
        loss_array.append(current_est)
        p_array.append(input_probs.detach().cpu().tolist())

        # 추정값과 입력 분포는 매 iteration 기록하고, 진행 상황만 20회마다 출력한다.
        if (epoch + 1) % 20 == 0:
            print(f"Run {run_index:02d}/{num_runs:02d} | Epoch {epoch+1:3d} "
                  f"| D_loss (Estimated U): {current_est:.5f}")

    return loss_array, np.asarray(p_array)


# ==========================================
# Repeated training & CSV updates
# ==========================================
# Figure에는 수치 reference의 중간값만 선으로 표시하고, 상·하한은 CSV에 보존한다.
df = pd.DataFrame({
    'Iteration': np.arange(1, epochs + 1),
    'Exact': exact_val,
    'Exact_Lower': exact_qui_result['lower'],
    'Exact_Upper': exact_qui_result['upper'],
    'MQUI_Lower': exact_mqui['lower'],
    'MQUI_Upper': exact_mqui['upper'],
    'MQUI_Midpoint': exact_mqui['midpoint'],
})
output_directory = Path(__file__).resolve().parents[1] / 'data' / 'cq6'
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / f'QUIGAN_cq_{N_X + N_B}qubits_seed{master_seed}.csv'
temporary_path = output_path.with_suffix('.csv.tmp')

for run_index in range(1, num_runs + 1):
    print(f"*** Starting run {run_index:02d}/{num_runs:02d} ***")
    estimates, p_history = train_once(run_index)
    df.insert(run_index, f'Estimation_{run_index:02d}', estimates)
    for x in range(num_inputs):
        df[f'p_{run_index:02d}_x{x}'] = p_history[:, x]

    # Replace the CSV only after the complete updated table has been written.
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    print(f"*** Run {run_index:02d}/{num_runs:02d} saved to {output_path} ***")


# ==========================================
# adjustable settings
# ==========================================
# p: 채널 입력 분포; P, Q: 측정 결과 분포.
# loss function: DV (Donsker-Varadhan) representation, L = E_P[h] - log(E_Q[exp(h)]).
# h의 zero-centering: h_i = w_params[i] - mean(w_params), 따라서 mean(h) = 0.
# 정규화된 P, Q에서 h 전체의 상수 이동은 DV 목적값을 바꾸지 않는다.
# input distribution: p를 직접 학습하고, 매 업데이트 후 probability simplex에 Euclidean projection.
# ansatz: hardware_efficient_ansatz (generator와 discriminator의 공통 회로 구조).
# 각 층에서 각 큐빗에 RX -> RY -> RZ를 적용한 뒤,
# 해당 회로의 wires 순서로 CNOT을 연결하고 마지막 -> 첫 번째까지 연결하는 ring 구조.
# layers_G = 8: generator 회로의 ansatz 층 수.
# layers_D = 256: discriminator 회로의 ansatz 층 수.
# N_anc_G = 3: generator가 sigma_B를 정화하여 준비할 때 사용하는 보조 큐빗 수.
# N_anc_D = 0: discriminator 측정 회로에 사용하는 보조 큐빗 수.
# epochs = 4000: 한 번의 학습에서 수행하는 총 iteration 수 (D/G/p 업데이트 묶음).
# num_runs = 10: 동일한 타깃에 대한 독립 학습 반복 횟수.
#
# 아래 reference 설정은 호출부와 각 모듈의 기본값에 지정되어 있다.
# reference tolerance = 1e-6: 입력 분포 최적화를 포함한 채널 값의 상·하한 절대 간격 허용치 (nats).
# exact max_retries = 3: 입력 분포 최적화를 최대 3회 추가 실행.
# measured max_retries = 3: 외부 입력 분포 최적화와 각 고정 p 내부 풀이에 각각 최대 3회 추가 시도.
# max_iterations 모듈 기본값: exact 200, measured 40; 각 외부 SLSQP 실행의 반복 한도.
#
# 아래 설정은 train_once 내부에 숫자로 직접 지정되어 있다.
# opt_G의 lr = 0.001: generator 파라미터 theta의 Adam 학습률.
# opt_phi의 lr = 0.001: discriminator 회로 파라미터 phi의 Adam 학습률.
# opt_w의 lr = 0.05: 측정 결과별 고전 파라미터 w_params의 Adam 학습률.
# lr_p = 0.001: 입력 확률 p의 projected gradient ascent 학습률.
# Discriminator의 range(10): iteration당 phi와 w_params를 업데이트하는 횟수.
# Generator의 range(1): iteration당 theta를 업데이트하는 횟수.
# Input distribution의 range(1): iteration당 p를 업데이트하고 simplex에 투영하는 횟수.

# Round 1 common initialization: G standard deviation 1.0, D 0.1, h = 0; CQ p uniform.
# All Adam learning rates (and CQ p step size) follow the half-flat/half-cosine schedule above.
