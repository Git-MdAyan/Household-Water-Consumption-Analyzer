import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ==========================================
# 1. SYNTHETIC DATA GENERATOR (Simulated Meter Logs)
# ==========================================
def generate_sample_data(days=90):
    np.random.seed(42)
    start_time = datetime(2025, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(days * 24)]
    
    flow_liters = []
    categories = []
    
    for dt in timestamps:
        hour = dt.hour
        # Diurnal consumption behavior (peaks in morning 7-9 AM, evening 7-9 PM)
        if 7 <= hour <= 9:
            base = np.random.uniform(25, 45)
            cat = np.random.choice(['Shower', 'Kitchen'], p=[0.7, 0.3])
        elif 19 <= hour <= 21:
            base = np.random.uniform(20, 40)
            cat = np.random.choice(['Kitchen', 'Laundry', 'Shower'], p=[0.4, 0.4, 0.2])
        elif 12 <= hour <= 14:
            base = np.random.uniform(10, 20)
            cat = 'Kitchen'
        elif 1 <= hour <= 5:
            # Baseline night hours (normal is near zero)
            base = np.random.uniform(0, 0.5)
            cat = 'None'
        else:
            base = np.random.uniform(3, 12)
            cat = np.random.choice(['Gardening', 'Kitchen', 'Laundry'], p=[0.3, 0.4, 0.3])
            
        # Simulate an ongoing pipeline leak during days 40 to 45 (between 2 AM and 5 AM)
        if 40 <= (dt - start_time).days <= 45 and 2 <= hour <= 5:
            base += 7.5  # Continuous 7.5 L/hr leak trickle
            cat = 'Leakage'
            
        flow_liters.append(round(base, 2))
        categories.append(cat)
        
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Flow_Liters': flow_liters,
        'Category': categories
    })
    
    df.to_csv('household_water_logs.csv', index=False)
    print("Dataset saved as 'household_water_logs.csv'.")
    return df


# ==========================================
# 2. ANALYTICS ENGINE & LEAK DETECTION
# ==========================================
class WaterConsumptionAnalyzer:
    def __init__(self, filepath, household_members=4):
        self.df = pd.read_csv(filepath)
        self.household_members = household_members
        self.preprocess_data()
        
    def preprocess_data(self):
        # Convert timestamp to datetime and extract time components
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Date'] = self.df['Timestamp'].dt.date
        self.df['Hour'] = self.df['Timestamp'].dt.hour
        self.df['DayOfWeek'] = self.df['Timestamp'].dt.day_name()
        
    def detect_night_leaks(self, threshold=5.0):
        """Identifies sustained off-peak flow between 1 AM and 5 AM."""
        night_mask = (self.df['Hour'] >= 1) & (self.df['Hour'] <= 5)
        night_df = self.df[night_mask]
        
        # Flag records exceeding safe baseline night threshold
        leaks = night_df[night_df['Flow_Liters'] > threshold]
        return leaks

    def get_summary_metrics(self):
        total_volume = self.df['Flow_Liters'].sum()
        days_count = self.df['Date'].nunique()
        avg_daily = total_volume / days_count
        per_capita_daily = avg_daily / self.household_members
        who_benchmark = 135.0  # WHO baseline in Liters per person per day
        
        return {
            "Total Volume (L)": round(total_volume, 2),
            "Daily Average (L)": round(avg_daily, 2),
            "Per Capita Daily (L)": round(per_capita_daily, 2),
            "Benchmark (L)": who_benchmark,
            "Excess Usage (L)": max(0, round(per_capita_daily - who_benchmark, 2))
        }

    # ==========================================
    # 3. VISUALIZATION DASHBOARD GENERATION
    # ==========================================
    def plot_dashboard(self):
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(14, 9))
        
        # Grid layout: 2 rows, 2 columns (Top spans across 2 columns)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        
        # --- SUBPLOT 1: Hourly Demand Profile & Leak Alert ---
        ax1 = fig.add_subplot(gs[0, :])
        hourly_pattern = self.df.groupby('Hour')['Flow_Liters'].mean().reset_index()
        
        ax1.plot(hourly_pattern['Hour'], hourly_pattern['Flow_Liters'], 
                 color='#1f77b4', marker='o', linewidth=2.5, label='Hourly Mean Flow (L)')
        ax1.axhline(y=15, color='red', linestyle='--', linewidth=1.5, label='Peak Threshold Limit')
        
        # Highlight Night Leak Detection Window (1 AM to 5 AM)
        ax1.axvspan(1, 5, color='orange', alpha=0.25, label='Nocturnal Window (Leak Monitoring)')
        ax1.text(3, 16, 'NIGHTTIME LEAK ALERT ZONE', color='darkred', 
                 fontweight='bold', ha='center', fontsize=10)
        
        ax1.set_title('Hourly Water Consumption Pattern & Leak Detection Window', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Hour of the Day (24-Hour Format)')
        ax1.set_ylabel('Consumption (Liters)')
        ax1.set_xticks(range(0, 24))
        ax1.legend(loc='upper right')

        # --- SUBPLOT 2: Categorical End-Use Breakdown ---
        ax2 = fig.add_subplot(gs[1, 0])
        valid_cats = self.df[self.df['Category'] != 'None']
        cat_data = valid_cats.groupby('Category')['Flow_Liters'].sum()
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        ax2.pie(cat_data, labels=cat_data.index, autopct='%1.1f%%', 
                startangle=140, colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
        ax2.set_title('Consumption Breakdown (90-Day Summary)', fontsize=12, fontweight='bold')

        # --- SUBPLOT 3: Benchmark Comparison ---
        ax3 = fig.add_subplot(gs[1, 1])
        metrics = self.get_summary_metrics()
        labels = ['Household\nPer-Capita', 'WHO Standard\nBenchmark']
        values = [metrics['Per Capita Daily (L)'], metrics['Benchmark (L)']]
        bar_colors = ['#e74c3c' if values[0] > values[1] else '#2ecc71', '#3498db']
        
        bars = ax3.bar(labels, values, color=bar_colors, width=0.45, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{height:.1f} L', ha='center', va='bottom', fontweight='bold')
                     
        ax3.set_ylim(0, max(values) + 30)
        ax3.set_ylabel('Liters per Person per Day')
        ax3.set_title('Household Usage vs. Conservation Benchmark', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('water_consumption_dashboard.png', dpi=300)
        print("Dashboard graphic successfully saved as 'water_consumption_dashboard.png'.")
        plt.show()


# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == '__main__':
    # Step 1: Generate sample log file
    generate_sample_data(days=90)
    
    # Step 2: Initialize Analyzer
    analyzer = WaterConsumptionAnalyzer('household_water_logs.csv', household_members=4)
    
    # Step 3: Print Statistical Report
    metrics = analyzer.get_summary_metrics()
    print("\n" + "="*45)
    print(" HOUSEHOLD WATER CONSUMPTION REPORT ")
    print("="*45)
    for k, v in metrics.items():
        print(f" {k:<28}: {v}")
    print("="*45)
    
    # Step 4: Check for leaks
    leaks = analyzer.detect_night_leaks(threshold=5.0)
    print(f"\n[ALERT STATUS]: {len(leaks)} abnormal nighttime leakage instances detected!")
    if not leaks.empty:
        print("Sample Anomaly Timestamps:")
        print(leaks[['Timestamp', 'Flow_Liters', 'Category']].head())
    
    # Step 5: Render and Export Dashboard
    analyzer.plot_dashboard()