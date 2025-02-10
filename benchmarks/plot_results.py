import numpy as np
import matplotlib.pyplot as plt

def load_data(log,sel_text='Training'):
    epoach,steps,loss,time,accuracy=[],[],[],[],[]
    with open(log,'r') as f:
        for line in f:
            if line.strip() and sel_text in line:
                line=line.strip().split(',')
                r1,r2,r3,r4,r5=float(line[1].split()[-1]),float(line[3].split()[-1]),float(line[4].split()[-1]),float(line[6].split()[-2]),float(line[7].split()[-1])
                epoach.append(r1)
                steps.append(r2)
                loss.append(r3)
                time.append(r4)
                accuracy.append(r5)
    return epoach, steps, loss, time, accuracy

def plot_metrics(models=['segment','spline','mlp','fourier'],\
                 metrics='loss',\
                 sel_text='Training',\
                 out_figure='loss'):

    label_map={'mlp':'MLP','fourier':'KAN-Fourier','segment':'KAN-Segment','spline':'KAN-Bspline'}

    plt.figure()
    plt.subplots_adjust(left=0.15,bottom=0.16,right=0.98,top=0.98)
    cs=['r','orange','cyan','lime']

    for j,model in enumerate(models):
        data=[]
        for i in range(10):
            log=f'{model}_{i:d}_log'
            eoach,steps,loss,time,accuracy=load_data(log,sel_text)
            if metrics=='loss':
                data.append(loss)
            if metrics=='accuracy':
                data.append(accuracy)
        data=np.array(data)
        steps=np.array(steps)
        data_mean,data_std=np.mean(data,axis=0),np.std(data,axis=0)
        plt.plot(steps,data_mean,c=cs[j],label=label_map[model])
        plt.fill_between(steps,data_mean-data_std*0.5,data_mean+data_std*0.5,fc=cs[j],alpha=0.5)

    plt.legend()
    plt.xlabel('Steps')
    if metrics=='accuracy':
        plt.ylim([0,1])
    plt.ylabel('Loss' if metrics=='loss' else 'Accuracy')
    plt.savefig(f'{out_figure}.jpg',dpi=600)


def plot_time(models=['mlp','segment','fourier','spline'],\
              sel_text='Training',\
              out_figure='time'):

    label_map={'mlp':'MLP','segment':'KAN-segment','fourier':'KAN-fourier','spline':'KAN-Bspline'}

    plt.figure()
    plt.subplots_adjust(left=0.19,bottom=0.16,right=0.98,top=0.98)
    cs=['cyan','r','lime','orange']
    times=[]
    for j,model in enumerate(models):
        data=[]
        for i in range(10):
            log=f'{model}_{i:d}_log'
            eoach,steps,loss,time,accuracy=load_data(log,sel_text)
            data.append(time)
        data=np.array(data)/10
        data_mean,data_std=np.mean(data),np.std(data)
        times.append([data_mean,data_std])
    times=np.array(times)
    plt.bar(range(len(models)),times[:,0],tick_label=[label_map[k] for k in models],yerr=times[:,1],fc='none',edgecolor=cs,hatch='///',ecolor=cs)
    plt.xticks(rotation=15)
    plt.ylabel('Time (Sec./step)')
    plt.savefig(f'{out_figure}.jpg',dpi=600)

if __name__=='__main__':
    params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'mathtext.fontset':'cm',
    'figure.figsize': [7.56/2.54,6.41/2.54],
    'font.family':'Times New Roman'
    }
    plt.rcParams.update(params)
    
    plot_metrics(metrics='loss',sel_text='Training',out_figure='loss_training')
    plot_metrics(metrics='loss',sel_text='Validation',out_figure='loss_validation')

    plot_metrics(metrics='accuracy',sel_text='Training',out_figure='accuracy_training')
    plot_metrics(metrics='accuracy',sel_text='Validation',out_figure='accuracy_validation')
    
    plot_time(sel_text='Training',out_figure='time_training')
    plot_time(sel_text='Validation',out_figure='time_validation')
