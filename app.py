import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random, time

# ==================== PENGATURAN HALAMAN ====================
st.set_page_config(page_title="S-Box GA Generator", layout="wide")
st.title("🧬 Konstruksi S-Box pada GF(2^8) dengan Genetic Algorithm")
st.markdown("""
Aplikasi ini mengoptimalkan matriks transformasi *Affine* untuk membentuk S-Box baru menggunakan Algoritma Genetika. 
Tujuannya adalah meminimalkan deviasi **Strict Avalanche Criterion (SAC)**.
""")

# ==================== FUNGSI KRIPTOGRAFI DASAR ====================
@st.cache_data # Menggunakan cache agar tidak membebani komputasi ulang jika tidak perlu
def gf_mul(a, b):
    p = 0
    for _ in range(8):
        if b & 1: p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi: a ^= 0x1B
        b >>= 1
    return p

@st.cache_data
def gf_inv(a):
    if a == 0: return 0
    for i in range(256):
        if gf_mul(a, i) == 1:
            return i
    return 0

def affine(x, M, c):
    bits = [(x >> i) & 1 for i in range(8)]
    out = 0
    for i in range(8):
        bit = sum(M[i][j] * bits[j] for j in range(8)) & 1
        out |= (bit << i)
    return out ^ c

def build_sbox(M, c):
    return [affine(gf_inv(x), M, c) for x in range(256)]

def make_aes_sbox():
    s = []
    for i in range(256):
        inv = gf_inv(i)
        c = 0x63
        res = inv
        res ^= (res << 1) ^ (res << 2) ^ (res << 3) ^ (res << 4)
        res = (res ^ (res >> 4) ^ c) & 0xFF
        s.append(res)
    return s

# ==================== FUNGSI EVALUASI ====================
def walsh_hadamard(f):
    wht = np.array([1 - 2*f(x) for x in range(256)], dtype=int)
    step = 1
    while step < 256:
        for i in range(0, 256, 2*step):
            for j in range(step):
                u, v = wht[i+j], wht[i+j+step]
                wht[i+j] = u + v
                wht[i+j+step] = u - v
        step <<= 1
    return wht

def nonlinearity(sbox):
    nl_min = 256
    for b in range(1, 256):
        def comp_func(x): return bin(sbox[x] & b).count('1') % 2
        wht = walsh_hadamard(comp_func)
        max_wht = np.max(np.abs(wht))
        nl = 128 - max_wht // 2
        if nl < nl_min: nl_min = nl
    return nl_min

def sac_analysis(sbox):
    n = 8
    mat = np.zeros((n, n))
    for i in range(256):
        orig = sbox[i]
        for ibit in range(n):
            flipped = i ^ (1 << ibit)
            newval = sbox[flipped]
            xor = orig ^ newval
            for obit in range(n):
                if xor & (1 << obit):
                    mat[ibit][obit] += 1
    mat /= 256.0
    return np.mean(mat), np.mean(np.abs(mat - 0.5)), mat

def bic_analysis(sbox):
    n = 8
    nl_vals, sac_vals = [], []
    for i in range(n):
        for j in range(i+1, n):
            mask_i, mask_j = 1 << i, 1 << j
            def comp_func(x):
                val = sbox[x]
                return ((val & mask_i) >> i) ^ ((val & mask_j) >> j)
            wht = walsh_hadamard(comp_func)
            max_wht = np.max(np.abs(wht))
            nl_vals.append(128 - max_wht // 2)
            mat = np.zeros(8)
            for x in range(256):
                orig = comp_func(x)
                for ibit in range(8):
                    if orig != comp_func(x ^ (1 << ibit)):
                        mat[ibit] += 1
            mat /= 256.0
            sac_vals.append(np.mean(mat))
    return np.mean(nl_vals), np.mean(sac_vals)

def lap(sbox):
    lap_max = 0.0
    for a in range(1, 256):
        for b in range(1, 256):
            def func(x):
                inp = bin(x & a).count('1') % 2
                out = bin(sbox[x] & b).count('1') % 2
                return inp ^ out
            wht = walsh_hadamard(func)
            max_abs = np.max(np.abs(wht[1:]))
            prob = (max_abs / 256.0) ** 2
            if prob > lap_max: lap_max = prob
    return lap_max

def dap(sbox):
    table = np.zeros((256, 256), dtype=int)
    for x in range(256):
        for dx in range(1, 256):
            dy = sbox[x] ^ sbox[x ^ dx]
            table[dx][dy] += 1
    return np.max(table[1:, :]) / 256.0

# ==================== FUNGSI ALGORITMA GENETIKA ====================
def random_invertible_matrix():
    while True:
        M = [[random.randint(0,1) for _ in range(8)] for _ in range(8)]
        if is_invertible(M):
            return M

def is_invertible(M):
    A = [row[:] for row in M]
    n = 8
    for col in range(n):
        row = None
        for r in range(col, n):
            if A[r][col] == 1:
                row = r
                break
        if row is None: return False
        A[col], A[row] = A[row], A[col]
        for r in range(n):
            if r != col and A[r][col] == 1:
                for c in range(n):
                    A[r][c] ^= A[col][c]
    return True

def individual_from_bits(bits):
    M = []
    for i in range(8):
        row = bits[i*8:(i+1)*8]
        M.append(row)
    c = int("".join(str(b) for b in bits[64:72]), 2)
    return M, c

def bits_from_individual(M, c):
    bits = []
    for row in M: bits.extend(row)
    c_bits = [(c >> i) & 1 for i in range(8)]
    bits.extend(c_bits)
    return bits

def fitness_fast(M, c):
    sbox = build_sbox(M, c)
    if len(set(sbox)) != 256: return -1000
    _, dev_sac, _ = sac_analysis(sbox)
    return -dev_sac

def crossover(bits1, bits2):
    point = random.randint(1, 70)
    return bits1[:point] + bits2[point:]

def mutate(bits, rate):
    for i in range(len(bits)):
        if random.random() < rate:
            bits[i] ^= 1

def evaluate_sbox_full(sbox):
    nl = nonlinearity(sbox)
    avg_sac, dev_sac, mat = sac_analysis(sbox)
    bic_nl, bic_sac = bic_analysis(sbox)
    lap_val = lap(sbox)
    dap_val = dap(sbox)
    return {
        "NL": nl,
        "SAC Avg": round(avg_sac, 4),
        "SAC Dev": round(dev_sac, 4),
        "BIC-NL": round(bic_nl, 2),
        "BIC-SAC": round(bic_sac, 4),
        "LAP": round(lap_val, 6),
        "DAP": round(dap_val, 6)
    }, mat

# ==================== UI STREAMLIT ====================
# Sidebar untuk parameter input
st.sidebar.header("⚙️ Parameter Algoritma Genetika")
POP_SIZE = st.sidebar.number_input("Ukuran Populasi", min_value=10, max_value=100, value=20, step=10)
GENERATIONS = st.sidebar.number_input("Jumlah Generasi", min_value=5, max_value=100, value=15, step=5)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.10, 0.03, 0.01)

if st.sidebar.button("🚀 Mulai Evolusi S-Box"):
    
    # Elemen untuk menunjukkan progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start = time.time()
    
    # 1. Inisialisasi Populasi
    status_text.text("Menginisialisasi populasi acak (matriks invertible)...")
    pop_bits = []
    for _ in range(POP_SIZE):
        while True:
            M = random_invertible_matrix()
            c = random.randint(0, 255)
            bits = bits_from_individual(M, c)
            M_check, _ = individual_from_bits(bits)
            if is_invertible(M_check):
                pop_bits.append(bits)
                break
                
    # 2. Loop Evolusi
    best_fit_history = []
    for gen in range(GENERATIONS):
        status_text.text(f"Sedang memproses Generasi {gen+1} dari {GENERATIONS}...")
        
        fits = []
        for bits in pop_bits:
            M, c = individual_from_bits(bits)
            fits.append(fitness_fast(M, c))
            
        best_idx = np.argmax(fits)
        best_fit_history.append(fits[best_idx])
        
        # Elitisme
        elite_idx = np.argsort(fits)[-2:]
        new_pop = [pop_bits[i][:] for i in elite_idx]
        
        # Crossover & Mutasi
        while len(new_pop) < POP_SIZE:
            t1, t2 = random.sample(range(POP_SIZE), 2)
            parent1 = pop_bits[t1] if fits[t1] > fits[t2] else pop_bits[t2]
            
            t1, t2 = random.sample(range(POP_SIZE), 2)
            parent2 = pop_bits[t1] if fits[t1] > fits[t2] else pop_bits[t2]
            
            child = crossover(parent1, parent2)
            mutate(child, MUTATION_RATE)
            
            M_child, c_child = individual_from_bits(child)
            if is_invertible(M_child):
                new_pop.append(child)
            else:
                new_pop.append(parent1[:])
                
        pop_bits = new_pop
        # Update progress bar
        progress_bar.progress((gen + 1) / GENERATIONS)

    # 3. Evaluasi Akhir
    status_text.text("Evolusi selesai! Mengevaluasi hasil akhir (menghitung LAP & DAP)...")
    
    final_fits = []
    for bits in pop_bits:
        M, c = individual_from_bits(bits)
        final_fits.append(fitness_fast(M, c))
    
    best_idx = np.argmax(final_fits)
    M_opt, c_opt = individual_from_bits(pop_bits[best_idx])
    opt_sbox = build_sbox(M_opt, c_opt)
    
    # Ambil nilai evaluasi
    ga_metrics, mat_ga = evaluate_sbox_full(opt_sbox)
    aes_metrics, mat_aes = evaluate_sbox_full(make_aes_sbox())
    
    time_taken = time.time() - start
    status_text.success(f"Evolusi selesai dalam {time_taken:.2f} detik!")
    
    # ==================== TAMPILAN HASIL ====================
    st.markdown("---")
    st.subheader("📊 Perbandingan Hasil Evaluasi")
    
    col1, col2 = st.columns(2)
    
    # Menampilkan tabel perbandingan
    with col1:
        st.markdown("**S-Box GA (Baru)**")
        st.json(ga_metrics)
    with col2:
        st.markdown("**S-Box AES (Referensi)**")
        st.json(aes_metrics)

    # Visualisasi Heatmap Matplotlib
    st.markdown("---")
    st.subheader("🔥 Visualisasi Matriks SAC (Strict Avalanche Criterion)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, mat, title in zip(axes, [mat_aes, mat_ga], ["SAC - AES", "SAC - GA Affine"]):
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(8), [f'b{i}' for i in range(8)])
        ax.set_yticks(range(8), [f'In{i}' for i in range(8)])
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
                
    fig.colorbar(im, ax=axes, label='Probabilitas')
    st.pyplot(fig) # Mengirim figure matplotlib ke Streamlit

    # Menampilkan HEX
    st.markdown("---")
    st.subheader("💻 Hasil S-Box GA (Format Hex)")
    hex_str = ""
    for i in range(0, 256, 16):
        hex_str += " ".join(f"{opt_sbox[i+j]:02X}" for j in range(16)) + "\n"
    st.code(hex_str, language="text")

else:
    st.info("Silakan atur parameter di sidebar dan klik **Mulai Evolusi S-Box**.")