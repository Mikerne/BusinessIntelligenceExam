import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# ----------------------------
# 1. Boxplots
# ----------------------------
def show_boxplots(df: pd.DataFrame, layout: str = "separate"):
    numeric_cols = df.select_dtypes(include="number").columns

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot for {col}')
            axes[i].set_ylabel(col)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        for column in numeric_cols:
            plt.figure(figsize=(6, 4))
            df.boxplot(column=column)
            plt.title(f'Boxplot for {column}')
            plt.ylabel(column)
            plt.tight_layout()
            plt.show()
    return fig, axes

# ----------------------------
# 2. Histograms
# ----------------------------
def show_histograms(df: pd.DataFrame, bins: int = 10, layout: str = "separate", bell_curve: bool = False):
    numeric_cols = df.select_dtypes(include="number").columns

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            data_df = df[col].dropna()
            axes[i].hist(data_df, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve:
                mu, std = data_df.mean(), data_df.std()
                xmin, xmax = axes[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                axes[i].plot(x, p, 'r', linewidth=2)
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.25)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            data_df = df[col].dropna()
            plt.hist(data_df, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve:
                mu, std = data_df.mean(), data_df.std()
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                plt.plot(x, p, 'r', linewidth=2)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()
    return fig, axes

# ----------------------------
# 3. Scatter Matrix
# ----------------------------
def show_scatter_matrix(df: pd.DataFrame, figsize: tuple = (12, 12), diagonal: str = "hist"):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Scatter matrix requires at least two numeric columns.")
        return
    scatter_matrix(df[numeric_cols], figsize=figsize, diagonal=diagonal)
    plt.suptitle("Scatter Matrix of Numeric Features")
    plt.show()

# ----------------------------
# 4. Correlation Heatmap
# ----------------------------
def show_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Correlation heatmap requires at least two numeric columns.")
        return
    corr = df[numeric_cols].corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()
    return fig

# ----------------------------
# 5. Grouped Histograms: Percent CHD for every bin
# ----------------------------
def show_grouped_histograms(df: pd.DataFrame, 
                            bins: int = 10, 
                            layout: str = "separate",
                            category_col: str = "TenYearCHD",
                            max_cols: int = 3,
                            alpha: float = 0.7):
    """
    Shows the percentage of people with CHD=1 per bin by continuos variables.
    x axis is the binned column
    y axix is percentage with CHD=1
    """
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame.")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != category_col]
    if not numeric_cols:
        print("No numeric columns to plot.")
        return

    def _plot_column(ax, col: str):
        # Lav bins
        bin_labels = pd.cut(df[col], bins=bins)
        # Calculate percentage for every bin
        pct_chd = df.groupby(bin_labels)[category_col].mean() * 100
        
        # Plot
        ax.bar(pct_chd.index.astype(str), pct_chd.values, alpha=alpha, color='salmon', edgecolor='black')
        ax.set_title(f'Procent med CHD per bin af {col}')
        ax.set_xlabel(f'{col} bins')
        ax.set_ylabel('Percentage with CHD (%)')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Layout
    if layout == "grid":
        n = len(numeric_cols)
        cols = max(1, min(max_cols, n))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
        flat_axes = axes.flatten()
        for ax, col in zip(flat_axes, numeric_cols):
            _plot_column(ax, col)
        for i in range(len(numeric_cols), len(flat_axes)):
            fig.delaxes(flat_axes[i])
        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_column(ax, col)
            plt.tight_layout()
            plt.show()





# ----------------------------
# 6. Binned Data
# ----------------------------
def show_binned_data(
    df: pd.DataFrame,
    bins: int = 5,
    column_to_bin: str = 'pH',
    column_to_plot: str = 'density',
    aggregation_method: str = 'mean'
):
    if column_to_bin not in df.columns or column_to_plot not in df.columns:
        raise ValueError("Specified columns must be present in the DataFrame.")

    binned_df = df[[column_to_bin, column_to_plot]].copy()
    binned_df['bin'] = pd.cut(binned_df[column_to_bin], bins=bins)

    if aggregation_method == 'max':
        binned_vals = binned_df.groupby('bin')[column_to_plot].max()
        agg_label = 'Max'
    else:
        binned_vals = binned_df.groupby('bin')[column_to_plot].mean()
        agg_label = 'Mean'

    plt.figure(figsize=(10, 6))
    binned_vals.plot(kind='bar', color='skyblue')
    plt.title(f'{agg_label} {column_to_plot} by {column_to_bin} Bin')
    plt.xlabel(f'{column_to_bin} Bin')
    plt.ylabel(f'{agg_label} {column_to_plot}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.show()
    return binned_vals
