from functions import *
import pandas as pd
import numpy as np
import os

def saveFileFigures(fig,directory,namefile):
    directory=directory+"figures/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory)
    fig.savefig(directory+namefile+".pdf")   # save the figure to file
    #plt.show()


#Prints on a file the big matrix (asked by professor)
def printBigPlot(directory,data,figsize,namefile,colors,cases):
    print("Printing Big Plot for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )

    for i in range(len(data)):
        for j in range(len(data[i])):
        #print(i,j)
            ax=axs[i][j]
            d=data[i][j].pkts["rtt"]
            ax.set_ylabel("Density")
            ax.set_title("Node "+ str(j) )
            ax.set_xlabel("Time (ms)")
            if not d.empty  | len(d)<2 :
                d.plot.kde(
                    ax=ax,
                    label="Case " +str(cases[i]),
                    color=colors[i]

                )


                d.hist(density=True,alpha=0.3,color=colors[i], ax=ax)

                ax.legend()
            #ax.set_xlim([-500, 8000])
    saveFileFigures(fig,directory,namefile)

#Print on a file density by Hop (asked by professor)
def printDensityByHop(directory,data,figsize,namefile,colors,cases):

    print("Printing Density by Hop for "+directory)
    dataHop=hopPreparation(data)
    fig, axs= plt.subplots(len(dataHop[0]),1, figsize=(15,20),sharey=True, )
    #print(len(dataHop),len(dataHop[0]))
    for i in range(len(dataHop)):
        for j in range(len(dataHop[i])):
            #print(i,j)
            d=dataHop[i][j]['rtt']
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_title("Hop "+ str(j+1))
            if not d.empty | len(d)<2 :
                d.plot.kde(
                    ax=axs[j],
                    label=cases[i],color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])


                axs[j].legend()

            #axs[j].set_xlim([-40, 6000])
    saveFileFigures(fig,directory,namefile)

#Print on a file density by Case (asked by professor)
def printDensityByCase(directory,data,figsize,namefile,colors,cases):

    print("Printing Density by case for "+directory)
    #print(len(data),len(data[0]))

    data1=hopPreparation(data)
    dataHopT=[*zip(*data1)]

    #print(len(data1),len(data1[0]))
    #print(len(dataHopT),len(dataHopT[0]))
    fig, axs= plt.subplots(len(dataHopT[0]),1, figsize=(15,20),sharey=True, )
    for i in range(len(dataHopT)):
        for j in range(len(dataHopT[0])):
            d=dataHopT[i][j]["rtt"]
            axs[j].set_title(""+ cases[i])
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_ylabel("Density")
            if not d.empty | len(d)<2 :
                d.plot.kde(
                    ax=axs[j],
                    label="Hop "+str(i),
                    color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])

                axs[j].legend()

            #axs[j].set_xlim([-40, 6000])
    saveFileFigures(fig,directory,namefile)

#Print Density of delay without outliers in every node by Case
def densityOfDelayByCaseNoOutliers(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay without outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getStdValues(data[i][j].pkts)
            if not out.empty :
                ax=axs[j]
                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                     color=colors[i]
            )
                ax.set_ylabel("Density")
                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.set_title("Node "+ str(j))
                ax.set_xlabel("Time (ms)")
                ax.legend()
    saveFileFigures(fig,directory,namefile)

#Density of outliers in every node by Case
def densityOutliersByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getOutliers(data[i][j].pkts)
            ax=axs[i][j]
            ax.set_ylabel("Density")
            ax.set_title("Node "+ str(j))
            ax.set_xlabel("Time (ms)")
            if not out.empty | len(out)<2 :

                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                 color=colors[i]
            )

                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.legend()


    saveFileFigures(fig,directory,namefile)


#Distibution of the delay divided by Node in the differents Cases
def densityOfDelayByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            d=data[i][j].pkts["rtt"]
            axs[j].set_title("Node "+ str(j))
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_ylabel("Density")
            if not d.empty | len(d)<2 :


                d.plot.kde(
                    ax=axs[j],
                    label=cases[i],color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])

                axs[j].legend()
    saveFileFigures(fig,directory,namefile)


#RTT Graph
def RTTGraph(directory,data,figsize,namefile,colors,cases):
    print("Printing RTT Graph for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            axs[j].plot(data[i][j].pkts["seq"],data[i][j].pkts["rtt"],label=cases[i],color=colors[i]   )
            axs[j].set_title("Node "+ str(j))
            axs[j].set_xlabel("Packet Number")
            axs[j].set_ylabel("Time (ms)")
            axs[j].legend()
    saveFileFigures(fig,directory,namefile)