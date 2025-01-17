import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

predictions_df = pd.read_csv("predictions_original_scale2.csv")  # First model
predictions_df_2 = pd.read_csv("/mnt/DATA/dawei/0_oak/predictions_original_scale2.csv")  # Second model
ground_truth_df = pd.read_csv("ground_truth_original_scale2.csv")

evaluation_results = {}

for col in predictions_df.columns:
    y_pred = predictions_df[col].values
    y_true = ground_truth_df[col].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    r2 = r2_score(y_true, y_pred)
    
    evaluation_results[col] = {"RMSE": rmse, "R²": r2}

results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')


results_df.to_csv("evaluation_metrics2.csv", index_label="Variable")
print("Evaluation metrics saved to 'evaluation_metrics2.csv'")
print(results_df)

print(f"Predictions rows: {len(predictions_df)}")
print(f"Predictions Model 2 rows: {len(predictions_df_2)}")
print(f"Ground Truth rows: {len(ground_truth_df)}")

variables = predictions_df_2.columns

error_thresholds = [0.05, 0.1, 0.15, 0.2]

for var in variables:
    predictions = predictions_df[var].values
    predictions_2 = predictions_df_2[var].values
    ground_truth = ground_truth_df[var].values

    m1, b1 = np.polyfit(ground_truth, predictions, 1)
    m2, b2 = np.polyfit(ground_truth, predictions_2, 1)

    thresholds_summary_1 = []
    thresholds_summary_2 = []
    for threshold in error_thresholds:
        lower_bound = ground_truth * (1 - threshold)
        upper_bound = ground_truth * (1 + threshold)
        
        within_threshold_1 = np.sum((predictions >= lower_bound) & (predictions <= upper_bound))
        percentage_within_threshold_1 = (within_threshold_1 / len(predictions)) * 100
        thresholds_summary_1.append(
            f"Training w/ updated loss function ({threshold * 100:.0f}%): {within_threshold_1}/{len(predictions)} ({percentage_within_threshold_1:.1f}%)"
        )
        
        within_threshold_2 = np.sum((predictions_2 >= lower_bound) & (predictions_2 <= upper_bound))
        percentage_within_threshold_2 = (within_threshold_2 / len(predictions_2)) * 100
        thresholds_summary_2.append(
            f"Training w/ original loss function ({threshold * 100:.0f}%): {within_threshold_2}/{len(predictions_2)} ({percentage_within_threshold_2:.1f}%)"
        )

    thresholds_legend = "\n".join(thresholds_summary_1 + thresholds_summary_2)

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        ground_truth,
        predictions,
        alpha=0.5,
        color="blue",
        label="Training w/ updated loss function"
    )
    plt.scatter(
        ground_truth,
        predictions_2,
        alpha=0.5,
        color="green",
        label="Training w/ original loss function"
    )

    # Perfect Prediction Line
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
        linestyle="--",
        label="Perfect Prediction Line"
    )
    
    # Regression Lines
    plt.plot(
        ground_truth,
        m1 * ground_truth + b1,
        color="blue",
        linestyle="-",
        label=f"Regression Line (Training w/ updated loss function): y = {m1:.2f}x + {b1:.2f}"
    )
    plt.plot(
        ground_truth,
        m2 * ground_truth + b2,
        color="green",
        linestyle="-",
        label=f"Regression Line (Training w/ original loss function): y = {m2:.2f}x + {b2:.2f}"
    )

    # Error Bounds for largest threshold
    largest_threshold = error_thresholds[-1]
    lower_bound = ground_truth * (1 - largest_threshold)
    upper_bound = ground_truth * (1 + largest_threshold)
    plt.fill_between(
        ground_truth,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.2,
        label=f"Error Bounds (up to {largest_threshold * 100:.0f}%)\n{thresholds_legend}"
    )

    # Add legend
    plt.legend(
        fontsize=12,
        loc="upper left"
    )

    plt.title(f"Scatter Plot for {var} - Training Comparison", fontsize=18)
    plt.xlabel("Ground Truth", fontsize=14)
    plt.ylabel("Predictions", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot instead of showing it
    plt.savefig(f"2_scatter_plot_comparison_{var}.png")
    plt.close()
