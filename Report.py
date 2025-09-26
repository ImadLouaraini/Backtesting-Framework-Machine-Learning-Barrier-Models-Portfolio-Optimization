import matplotlib.pyplot as plt

def plot_portfolio_results(results):
    plt.plot(results['dates'], results['portfolio_values'], label="Portfolio")
    plt.plot(results['dates'], results['benchmark_values'], label="Benchmark")
    plt.legend(); plt.show()

def print_report(results, metrics):
    print("Final Portfolio Value:", metrics['Final Value'])
    for k, v in metrics.items():
        if k != "Final Value":
            print(f"{k}: {v}")
