# For GA algorithm (Genetic Algorithm)
# Employee Shift Scheduling 

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="GA Scheduler", layout="wide")
st.title("🧬 Employee Shift Scheduling (Genetic Algorithm)")
st.markdown("""
**Unique Approach:** Uses **Integer Encoding** (0=Off, 1=Morning, 2=Evening) to guarantee 
consecutive 8-hour shifts, eliminating broken schedules by design.
""")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# Penalties (Soft Constraints)
PENALTY_SHORTAGE = 200     # Demand not met
PENALTY_OVERHOURS = 150    # Workload > Max
PENALTY_DAYS_MIN = 300     # Worked < 6 days
PENALTY_SHIFT_BREAK = 100  # Sanity check (rarely needed in GA)
PENALTY_NONCONSEC = 200    # Sanity check (rarely needed in GA)

# ==========================================
# 2. DATA LOADING
# ==========================================
# Initialize Demand Matrix: (Dept, Day, Period)
DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int) 
folder_path = "./Demand/" 

if not os.path.exists(folder_path):
    st.error(f"❌ Folder '{folder_path}' not found. Please create it and add Dept1.xlsx to Dept6.xlsx.")
else:
    for dept in range(n_departments):
        file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
        if not os.path.exists(file_path):
            st.sidebar.warning(f"⚠️ Dept{dept+1}.xlsx not found")
            continue
        try:
            df = pd.read_excel(file_path, header=None) 
            # Extract 7 days x 28 periods (skip headers)
            df_subset = df.iloc[1:1+n_days, 1:1+n_periods] 
            df_subset = df_subset.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            DEMAND[dept] = df_subset.values 
        except Exception as e:
            st.error(f"Error reading Dept{dept+1}: {e}")

# ==========================================
# 3. HELPER FUNCTIONS & FITNESS
# ==========================================
def longest_consecutive_ones(arr): 
    max_len = curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len

def compute_penalty_breakdown(schedule, demand, max_hours): 
    """
    Calculates detailed penalties for a binary schedule.
    """
    total_shortage = 0
    total_overwork = 0
    total_days_min = 0
    total_shift_break = 0
    total_nonconsec = 0

    n_depts, days, periods, employees = schedule.shape

    # 1. Demand Shortage
    for dept in range(n_depts):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept,d,t,:])
                required = demand[dept,d,t]
                if assigned < required:
                    total_shortage += (required - assigned) * PENALTY_SHORTAGE 

    # 2. Employee Constraints
    for dept in range(n_depts):
        for e in range(employees):
            # Check if employee exists (some columns might be padding)
            # In this GA, we assume fixed employees per dept, but matrix is max_emps wide.
            # Empty slots are all 0s, so they won't trigger overwork, but might trigger min_days.
            # We skip 'ghost' employees who have 0 hours total if they weren't supposed to exist,
            # but here we rely on the GA to just optimize valid slots.
            
            # Total Hours
            total_hours = np.sum(schedule[dept, :, :, e])
            if total_hours > 0: # Only check constraints if they worked at all
                if total_hours > max_hours:
                    total_overwork += (total_hours - max_hours) * PENALTY_OVERHOURS 

                # Min Working Days (Expect 6 days)
                days_worked = np.sum(np.sum(schedule[dept, :, :, e], axis=1) > 0)
                if days_worked < (n_days - 1):
                    total_days_min += PENALTY_DAYS_MIN 

                # Shift Consistency (Sanity check)
                for d in range(days):
                    daily = schedule[dept,d,:,e]
                    worked = np.sum(daily)
                    if worked > 0:
                        if worked != SHIFT_LENGTH: 
                            total_shift_break += PENALTY_SHIFT_BREAK 
                        if longest_consecutive_ones(daily) < SHIFT_LENGTH:
                            total_nonconsec += PENALTY_NONCONSEC 

    total_fitness = total_shortage + total_overwork + total_days_min + total_shift_break + total_nonconsec
    return {
        "total_fitness": total_fitness,
        "shortage": total_shortage,
        "overwork": total_overwork,
        "days_min": total_days_min,
        "shift_break": total_shift_break,
        "nonconsec": total_nonconsec
    }

def fitness_function(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total_fitness"]

# ==========================================
# 4. GA CORE LOGIC (INTEGER ENCODING)
# ==========================================

def decode_chromosome(chromosome, n_depts, n_days, n_periods, max_emps):
    """
    Decodes Integer Chromosome -> Binary Schedule
    0 = Off
    1 = Morning (0-14)
    2 = Evening (14-28)
    """
    schedule = np.zeros((n_depts, n_days, n_periods, max_emps), dtype=int)
    
    for dept in range(n_depts):
        for d in range(n_days):
            for e in range(max_emps):
                val = chromosome[dept, d, e]
                if val == 1:
                    schedule[dept, d, 0:SHIFT_LENGTH, e] = 1
                elif val == 2:
                    schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1
    return schedule

def create_individual(n_depts, n_days, n_emps_per_dept, max_emps, rest_prob):
    # Chromosome: (Dept, Day, Emp) -> Values 0, 1, 2
    chromosome = np.zeros((n_depts, n_days, max_emps), dtype=int)
    
    for dept in range(n_depts):
        n_emp = n_emps_per_dept[dept]
        for e in range(n_emp):
            for d in range(n_days):
                if random.random() < rest_prob:
                    chromosome[dept, d, e] = 0
                else:
                    chromosome[dept, d, e] = random.choice([1, 2])
    return chromosome

def crossover(parent1, parent2, crossover_rate):
    """Uniform Crossover"""
    if random.random() > crossover_rate:
        return parent1.copy()
    
    mask = np.random.rand(*parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

def mutate(individual, mutation_rate, n_emps_per_dept):
    """Random Reset Mutation"""
    mask = np.random.rand(*individual.shape) < mutation_rate
    
    # Randomly assign 0, 1, or 2
    random_moves = np.random.randint(0, 3, size=individual.shape)
    
    individual = np.where(mask, random_moves, individual)
    
    # Cleanup: Ensure ghost employees (beyond actual count) stay 0
    n_depts, _, max_emps = individual.shape
    for dept in range(n_depts):
        n_emp = n_emps_per_dept[dept]
        if n_emp < max_emps:
            individual[dept, :, n_emp:] = 0
            
    return individual

def GA_scheduler(demand, n_employees_per_dept, pop_size, n_generations,
                 crossover_rate, mutation_rate, max_hours, rest_prob, early_stop):
    
    start_time = time.time()
    max_emps = max(n_employees_per_dept)
    
    # 1. Init Population
    population = [create_individual(n_departments, n_days, n_employees_per_dept, max_emps, rest_prob) 
                  for _ in range(pop_size)]
    
    fitness_history = []
    best_score_global = float("inf")
    best_chromosome_global = None
    no_improve = 0
    
    progress_bar = st.progress(0)
    
    for gen in range(n_generations):
        # 2. Evaluate
        scores = []
        decoded_pop = []
        
        for ind in population:
            sched_binary = decode_chromosome(ind, n_departments, n_days, n_periods, max_emps)
            score = fitness_function(sched_binary, demand, max_hours)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Track Best
        min_idx = np.argmin(scores)
        current_best_score = scores[min_idx]
        current_best_ind = population[min_idx]
        
        if current_best_score < best_score_global:
            best_score_global = current_best_score
            best_chromosome_global = current_best_ind.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        fitness_history.append({
            "iteration": gen + 1,
            "best": current_best_score,
            "mean": np.mean(scores),
            "worst": np.max(scores)
        })
        
        # Update Progress Bar
        progress_bar.progress((gen + 1) / n_generations)
        
        # Early Stopping
        if no_improve >= early_stop:
            st.toast(f"🛑 Early stopping at Gen {gen+1}")
            break
            
        # 3. Selection (Tournament)
        new_population = []
        new_population.append(current_best_ind.copy()) # Elitism
        
        while len(new_population) < pop_size:
            # Tournament Size = 3
            candidates_idx = np.random.choice(len(population), 3, replace=False)
            best_cand_idx = candidates_idx[np.argmin(scores[candidates_idx])]
            parent1 = population[best_cand_idx]
            
            candidates_idx = np.random.choice(len(population), 3, replace=False)
            best_cand_idx = candidates_idx[np.argmin(scores[candidates_idx])]
            parent2 = population[best_cand_idx]
            
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate, n_employees_per_dept)
            new_population.append(child)
            
        population = new_population

    progress_bar.empty()
    
    # Decode final best for return
    best_schedule_binary = decode_chromosome(best_chromosome_global, n_departments, n_days, n_periods, max_emps)
    run_time = time.time() - start_time

    return best_schedule_binary, best_score_global, fitness_history, run_time, best_chromosome_global

# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================

# Sidebar
st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 10, 200, 50)
n_generations = st.sidebar.slider("Generations", 10, 500, 50)
early_stop = st.sidebar.slider("Early Stop Patience", 5, 50, 15)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.2, 0.05)
REST_PROB = st.sidebar.slider("Initial Rest Prob", 0.0, 0.8, 0.3)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

st.sidebar.markdown("---")
st.sidebar.header("Staffing Levels")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Staff", 1, 50, 20, key=f"d{i}") for i in range(n_departments)
]

# Run Button
if st.sidebar.button("Run Genetic Algorithm 🧬", type="primary"):
    
    with st.spinner("Evolving Schedules... Please wait."):
        # Run Algorithm
        best_sched, best_score, hist, rtime, best_genes = GA_scheduler(
            DEMAND, n_employees_per_dept, pop_size, n_generations,
            crossover_rate, mutation_rate, max_hours, REST_PROB, early_stop
        )
        
    # --- RESULTS SECTION ---
    
    # 1. Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Fitness (Penalty)", f"{best_score:.0f}")
    c2.metric("Generations Run", f"{len(hist)}")
    c3.metric("Compute Time", f"{rtime:.2f} s")

    # 2. Evolution Tunnel (Visualization 1)
    st.subheader("📈 Evolution Progress (Diversity Tunnel)")
    
    iters = [x["iteration"] for x in hist]
    best_vals = [x["best"] for x in hist]
    mean_vals = [x["mean"] for x in hist]
    
    fig_evo, ax_evo = plt.subplots(figsize=(10, 4))
    ax_evo.fill_between(iters, best_vals, mean_vals, color='skyblue', alpha=0.3, label="Diversity (Best to Avg)")
    ax_evo.plot(iters, best_vals, color='blue', linewidth=2, label="Best Fitness")
    ax_evo.plot(iters, mean_vals, color='orange', linestyle='--', label="Avg Fitness")
    ax_evo.set_xlabel("Generation")
    ax_evo.set_ylabel("Penalty Score")
    ax_evo.legend()
    ax_evo.grid(True, alpha=0.3)
    st.pyplot(fig_evo)

    # 3. Penalty Radar Chart (Visualization 2)
    st.subheader("🎯 Constraint Balance (Radar Chart)")
    bd = compute_penalty_breakdown(best_sched, DEMAND, max_hours)
    
    cats = ['Shortage', 'Overwork', 'Min Days', 'Shift Break', 'Consecutive']
    vals = [bd['shortage'], bd['overwork'], bd['days_min'], bd['shift_break'], bd['nonconsec']]
    
    # Normalize for visuals? Raw values are okay if similar magnitude.
    # If one is huge (e.g. 5000), it dwarfs others. Let's use Log scale or Raw. Sticking to Raw for honesty.
    
    # Radar Setup
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals += vals[:1]
    
    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, vals, linewidth=2, linestyle='solid', color='#E63946')
    ax_radar.fill(angles, vals, '#E63946', alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(cats)
    ax_radar.set_title("Penalty Distribution (Smaller shape is better)")
    st.pyplot(fig_radar)
    
    # 4. Gene Map (Visualization 3)
    st.subheader("🧬 Chromosome Inspection (The Gene Map)")
    st.info("Visualizing the raw Integer Encoding. Notice the solid blocks of Blue (Morning) and Orange (Evening).")
    
    dept_tabs = st.tabs([f"Dept {i+1}" for i in range(n_departments)])
    
    for i, tab in enumerate(dept_tabs):
        with tab:
            n_emp = n_employees_per_dept[i]
            # Extract genes: (Day, Emp) for this dept
            genes = best_genes[i, :, :n_emp].T # Transpose to get Emp (rows) x Days (cols)
            
            fig_gene, ax_gene = plt.subplots(figsize=(8, n_emp * 0.4 + 1))
            
            # Colormap: 0=Grey, 1=Blue, 2=Orange
            cmap = ListedColormap(['#f0f0f0', '#3498db', '#e67e22'])
            
            im = ax_gene.imshow(genes, cmap=cmap, aspect='auto', vmin=0, vmax=2)
            
            ax_gene.set_xticks(range(n_days))
            ax_gene.set_xticklabels([f"D{d+1}" for d in range(n_days)])
            ax_gene.set_yticks(range(n_emp))
            ax_gene.set_yticklabels([f"E{e+1}" for e in range(n_emp)])
            
            # Grid
            ax_gene.set_xticks(np.arange(-.5, n_days, 1), minor=True)
            ax_gene.set_yticks(np.arange(-.5, n_emp, 1), minor=True)
            ax_gene.grid(which='minor', color='white', linestyle='-', linewidth=2)
            ax_gene.tick_params(which='minor', bottom=False, left=False)
            
            # Custom Legend
            legend_elements = [
                Patch(facecolor='#f0f0f0', edgecolor='gray', label='Rest (0)'),
                Patch(facecolor='#3498db', edgecolor='gray', label='Morning (1)'),
                Patch(facecolor='#e67e22', edgecolor='gray', label='Evening (2)')
            ]
            ax_gene.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
            st.pyplot(fig_gene)
            
            # Summary Text
            total_shortage = 0
            # Quick calc for shortage
            for d in range(n_days):
                for t in range(n_periods):
                    req = DEMAND[i,d,t]
                    assigned = np.sum(best_sched[i,d,t,:])
                    if assigned < req: total_shortage += (req-assigned)
            st.caption(f"Dept {i+1} Total Shortage: {int(total_shortage)} people-periods")

    # 5. Tabular Data (Optional, for detail)
    with st.expander("See Detailed Tables"):
        shift_mapping = {"09:00-17:00": range(0, SHIFT_LENGTH),
                         "14:00-22:00": range(14, 14+SHIFT_LENGTH)}
        
        for dept in range(n_departments):
            st.write(f"**Department {dept+1}**")
            rows = []
            n_emp = n_employees_per_dept[dept]
            emp_ids = [f"E{x+1}" for x in range(n_emp)]
            
            for d in range(n_days):
                for shift_name, rng in shift_mapping.items():
                    assigned = []
                    for e in range(n_emp):
                        # check if assigned in this block (just check first hour)
                        if best_sched[dept, d, rng[0], e] == 1:
                            assigned.append(emp_ids[e])
                    
                    rows.append([f"Day {d+1}", shift_name, ", ".join(assigned)])
            
            df_sched = pd.DataFrame(rows, columns=["Day", "Shift", "Staff"])
            st.dataframe(df_sched, use_container_width=True)
