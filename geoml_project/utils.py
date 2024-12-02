import matplotlib.pyplot as plt
from config import OUT_DIR

def create_loss_plot(
        loss_data, 
        x_label, 
        y_label, 
        save_name, 
        show=False, 
        save=True, 
        DIR=OUT_DIR
    ):

    fig = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = fig.add_subplot()
    ax.plot(loss_data, color='tab:blue')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if save:
        fig.savefig(f'{DIR}/{save_name}_loss.png')
        print("Saved Loss Plot...")

    if show:
        fig.show()



def create_mAP_plot(
        data, 
        save_name, 
        show=False, 
        save=True, 
        DIR=OUT_DIR
    ):

    fig = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = fig.add_subplot()

    map50_data = data['map50']
    map_data = data['map']

    ax.plot(map50_data, color='tab:orange', linestyle='-', label='mAP@0.5')
    ax.plot(map_data, color='tab:red', linestyle='-', label='mAP@0.5:0.95')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')

    ax.legend()

    if save:
        fig.savefig(f'{DIR}/{save_name}_mAP.png')
        print("Saved mAP Plot...")

    if show:
        fig.show()