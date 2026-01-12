# For GA algorithm
# Employee Shift Scheduling 

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

#config
st.title(" Employee Shift Scheduling (GA) 🧬 ")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# Penalties
PENALTY_SHORTAGE = 200 
PENALTY_OVERHOURS = 150 
PENALTY_DAYS_MIN = 300 
PENALTY_SHIFT_BREAK = 100 
PENALTY_NONCONSEC = 200 

# LOAD DEMAND
DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int) 
folder_path = "./Demand/" 

# Ensure demand folder exists or handle error
if not os.path.exists(folder_path):
    st.error(f"❌ Folder '{folder_path}' not found. Please create it and add Dept1.xlsx to Dept6.xlsx.")
else:
    for dept in range(n_departments):
        file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
        if not os.path.exists(file_path):
            st.sidebar.warning(f"⚠️ Dept{dept+1}.xlsx not found")
            continue
        df = pd.read_excel(file_path, header=None) 
        df_subset = df.iloc[1:1+n_days, 1:1+n_periods] 
        df_subset = df_subset.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        DEMAND[dept] = df_subset.values 

# HELPER FUNCTIONS 

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
    total_shortage = 0
    total_overwork = 0
    total_days_min = 0
    total_shift_break = 0
    total_nonconsec = 0

    n_departments, days, periods, employees = schedule.shape

    for dept in range(n_departments):
        # 1. Demand Shortage
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept,d,t,:])
                required = demand[dept,d,t]
                if assigned < required:
                    total_shortage += (required - assigned) * PENALTY_SHORTAGE 

        # 2. Employee Constraints
        for e in range(employees):
            # Overwork
            total_hours = np.sum(schedule[dept, :, :, e]) # Note: In GA simplified view, we usually sum per dept if employees are unique to dept
            # However, adapting to input shape:
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                total_overwork += (total_hours - max_hours) * PENALTY_OVERHOURS 

            # Min Working Days (at least 6 days i.e. max 1 day off usually, or just check days worked)
            # The logic here checks if they worked < 6 days
            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked < (n_days - 1):
                total_days_min += PENALTY_DAYS_MIN 

        # 3. Shift Constraints (sanity check, though GA representation handles this)
        for d in range(days):
            for e in range(employees):
                daily = schedule[dept,d,:,e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH: 
                    total_shift_break += PENALTY_SHIFT_BREAK 
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
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

# Wrapper for single objective
def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total_fitness"]

# GA HELPER FUNCTIONS

def decode_chromosome(chromosome, n_depts, n_days, n_periods, max_emps):
    """
    Convert simplified GA chromosome (0,1,2) into full binary schedule.
    Chromosome shape: (n_depts, n_days, max_emps)
    Values: 0=Off, 1=Morning (0-14), 2=Evening (14-28)
    """
    schedule = np.zeros((n_depts, n_days, n_periods, max_emps), dtype=int)
    
    # Vectorized decoding is possible but keeping loops for clarity/safety with shapes
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
    # Initialize with random shifts (0, 1, or 2)
    # 0 = Off, 1 = Morning, 2 = Evening
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
    if random.random() > crossover_rate:
        return parent1.copy()
    
    # Uniform Crossover: each gene has 50% chance coming from P1 or P2
    mask = np.random.rand(*parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

def mutate(individual, mutation_rate, n_emps_per_dept):
    # Random Reset Mutation
    mask = np.random.rand(*individual.shape) < mutation_rate
    
    n_depts, n_days, max_emps = individual.shape
    
    # Generate random new moves for the whole matrix
    random_moves = np.random.randint(0, 3, size=individual.shape) # 0, 1, 2
    
    # Apply mutation only where mask is True
    individual = np.where(mask, random_moves, individual)
    
    # Clean up non-existent employees (if max_emps > actual emps)
    for dept in range(n_depts):
        n_emp = n_emps_per_dept[dept]
        if n_emp < max_emps:
            individual[dept, :, n_emp:] = 0
            
    return individual

def GA_scheduler(demand, n_employees_per_dept, pop_size, n_generations,
                 crossover_rate, mutation_rate, max_hours, rest_prob, early_stop):
    
    start_time = time.time()
    max_emps = max(n_employees_per_dept)
    
    # 1. Initialize Population
    population = [create_individual(n_departments, n_days, n_employees_per_dept, max_emps, rest_prob) 
                  for _ in range(pop_size)]
    
    fitness_history = []
    best_score_global = float("inf")
    best_chromosome_global = None
    no_improve = 0
    
    for gen in range(n_generations):
        # 2. Evaluate Fitness
        scores = []
        decoded_pop = []
        
        for ind in population:
            # Decode to binary schedule for fitness calculation
            sched_binary = decode_chromosome(ind, n_departments, n_days, n_periods, max_emps)
            score = fitness(sched_binary, demand, max_hours)
            scores.append(score)
            decoded_pop.append(sched_binary)
        
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
        
        # Early Stopping
        if no_improve >= early_stop:
            break
            
        # 3. Selection (Tournament)
        new_population = []
        # Elitism: keep the very best
        new_population.append(current_best_ind.copy())
        
        while len(new_population) < pop_size:
            # Tournament size 3
            candidates_idx = np.random.choice(len(population), 3, replace=False)
            best_cand_idx = candidates_idx[np.argmin(scores[candidates_idx])]
            parent1 = population[best_cand_idx]
            
            candidates_idx = np.random.choice(len(population), 3, replace=False)
            best_cand_idx = candidates_idx[np.argmin(scores[candidates_idx])]
            parent2 = population[best_cand_idx]
            
            # 4. Crossover
            child = crossover(parent1, parent2, crossover_rate)
            
            # 5. Mutation
            child = mutate(child, mutation_rate, n_employees_per_dept)
            
            new_population.append(child)
            
        population = new_population

    # Final Result
    best_schedule_binary = decode_chromosome(best_chromosome_global, n_departments, n_days, n_periods, max_emps)
    run_time = time.time() - start_time
    
    # Identify off days for visualization (0 in chromosome)
    # shape (n_dept, n_days, max_emp)
    # We want format: list of (n_emp, n_days) per dept
    best_off_schedules = []
    for dept in range(n_departments):
        n_emp = n_employees_per_dept[dept]
        off_matrix = np.zeros((n_emp, n_days), dtype=int)
        for e in range(n_emp):
            for d in range(n_days):
                if best_chromosome_global[dept, d, e] == 0:
                    off_matrix[e, d] = 1
        best_off_schedules.append(off_matrix)

    return best_schedule_binary, best_score_global, fitness_history, run_time, best_off_schedules

# STREAMLIT CONTROLS

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 10, 200, 50)
n_generations = st.sidebar.slider("Generations", 10, 500, 50)
early_stop = st.sidebar.slider("Early Stop Generations", 5, 50, 15)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.2, 0.05)
REST_PROB = st.sidebar.slider("Initial Rest Probability", 0.0, 0.8, 0.3)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

# Employees Setup
st.sidebar.header("Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees", 1, 50, 20) for i in range(n_departments)
]

# RUN GA
if st.sidebar.button("Run GA"):
    with st.spinner("Evolving schedules... 🧬"):
        best_schedule, best_score, fitness_history, run_time, best_off_schedules = \
            GA_scheduler(DEMAND, n_employees_per_dept, pop_size, n_generations,
                         crossover_rate, mutation_rate, max_hours, REST_PROB, early_stop)

    st.success(f"Best Fitness Score: {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # Fitness Convergence with "Evolutionary Cloud"
    iters = [x["iteration"] for x in fitness_history]
    best_vals = [x["best"] for x in fitness_history]
    mean_vals = [x["mean"] for x in fitness_history]
    worst_vals = [x["worst"] for x in fitness_history]

    st.subheader("🧬 Evolutionary Progress")
    fig, ax = plt.subplots()
    
    # Plot the "Cloud" (Diversity of Population)
    ax.fill_between(iters, best_vals, worst_vals, color='green', alpha=0.1, label="Population Spread")
    
    # Plot Lines
    ax.plot(iters, mean_vals, color='green', linestyle='--', alpha=0.6, label="Average Fitness")
    ax.plot(iters, best_vals, color='darkgreen', linewidth=2, label="Best Fitness")
    
    # Highlight Stop
    if len(iters) < n_generations:
        ax.axvline(iters[-1], color='red', linestyle=':', label="Early Stop")
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Penalty Score")
    ax.set_title("Genetic Evolution (Population Convergence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Fitness Breakdown
    st.subheader("Fitness Breakdown")
    breakdown = compute_penalty_breakdown(best_schedule, DEMAND, max_hours)
    st.json(breakdown)

    # DISPLAY SCHEDULE + HEATMAP
    st.subheader("Department Schedule & Heatmap")
    shift_mapping = {"09:00-17:00": range(0, SHIFT_LENGTH),
                     "14:00-22:00": range(14, 14+SHIFT_LENGTH)}

    summary_rows = []
    for dept in range(n_departments):
        n_emp = n_employees_per_dept[dept]
        employee_ids = [f"E{i+1}" for i in range(n_emp)]
        off_schedule = best_off_schedules[dept]

        st.markdown(f"### Department {dept+1}")
        rows = []
        heatmap_data = np.zeros((n_days, len(shift_mapping)))
        total_shortage_dept = 0

        for d in range(n_days):
            for idx, (shift_label, period_range) in enumerate(shift_mapping.items()):
                assigned_emps = set()
                shortage_total_shift = 0
                shortage_periods = {}

                for t in period_range:
                    if t >= n_periods: continue
                    assigned = [employee_ids[e] for e in range(n_emp) if best_schedule[dept,d,t,e]==1]
                    assigned_emps.update(assigned)
                    shortage = DEMAND[dept,d,t] - len(assigned)
                    if shortage > 0:
                        shortage_periods[f"P{t+1}"] = shortage
                        shortage_total_shift += shortage

                off_today = [employee_ids[e] for e in range(n_emp) if off_schedule[e,d]==1]
                heatmap_data[d, idx] = shortage_total_shift
                total_shortage_dept += shortage_total_shift

                rows.append([f"Day {d+1}", shift_label,
                             ", ".join(sorted(assigned_emps)) or "-",
                             ", ".join(off_today) or "-",
                             ", ".join([f"{k}({v})" for k,v in shortage_periods.items()]) or "-"])

        df_dept = pd.DataFrame(rows, columns=["Day","Shift","Employees Assigned","Employee Off","Shortage (People per Period)"])
        st.dataframe(df_dept.style.applymap(lambda v: "background-color:red;color:white" if v!="-"
                                            else "", subset=["Shortage (People per Period)"]),
                     use_container_width=True)

        st.markdown(f"**Total Shortage for Department {dept+1}: {total_shortage_dept} people**")
        summary_rows.append([f"Department {dept+1}", total_shortage_dept])

        # Heatmap
        st.markdown(f"Shortage Heatmap - Dept {dept+1}")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(heatmap_data, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(shift_mapping)))
        ax.set_xticklabels(list(shift_mapping.keys()))
        ax.set_yticks(range(n_days))
        ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
        for i in range(n_days):
            for j in range(len(shift_mapping)):
                ax.text(j,i,int(heatmap_data[i,j]),ha="center",va="center")
        st.pyplot(fig)

    st.subheader("Summary Total Shortage per Department")
    df_summary = pd.DataFrame(summary_rows, columns=["Department","Total Shortage (People)"])
    st.dataframe(df_summary,use_container_width=True)
