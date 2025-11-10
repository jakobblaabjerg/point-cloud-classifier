import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(dataset, sample_size=None, random_state=42, save=None):

    df = dataset.copy()

    if sample_size is not None:
        df = df.groupby("label", group_keys=False).sample(
            n=sample_size, random_state=random_state
        )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="energy_total", 
        y="hits_total", 
        hue="label", 
        alpha=0.7
    )
    plt.xlabel("Shower Energy (MeV)")
    plt.ylabel("Number of Hits")
    plt.legend(title="Particle")
    plt.grid(True)

    if save:
        if save is True:
            filename = "plot.png"
        else:
            filename = save 
        
        plt.savefig(filename)
        plt.close()   
    
    else:
        plt.show()



