def analyze_network(directory,plots,pings,window):
    cases=[]
    casesAccuracy=[]
    for row in plots:
        cases.append(row[1])
        casesAccuracy.append(row[2])
        data=import_Cooja2(plots)
    pings=getPings(data)
    #All data collection is in variable node that is a list of list of nodes
    #3 nets input x 9 nodes by net

    d={ "label":[],
       "type":[],
        "count":[],
        "std":  [],
        "mean": [],
        "var":  [],
        "hop":[],

       "packet loss":[],
       "outliers":[],
       "node":[]
    }
    #count=[]
    labels=[]
    var=[]
    #window=100
    #stats=pd.DataFrame(columns=columns)

    for i in range(len(data)):
        #window=pings[i]

        for j in range(len(data[i])):
            #n=pings[i]
            n=100
            window=50
            #print(n)
            for z in range(0,n,int(window)):
                #if(z+window>n):break
                #print(z,z+window)

                #df1 = df1.assign(e=p.Series(np.random.randn(sLength)).values)
                node=data[i][j].pkts
                name=str(j)+" "+cases[i]
                nodeWindow=node[(node["seq"]<z+window) & (node["seq"]>=z)]
                nodeWindowP=nodeWindow["rtt"]
                d["count"].append(nodeWindowP.count())
                #Case without outliers
                #Case with outliers
                std=0
                if (nodeWindowP.std()>10):
                    std=1
                    std=nodeWindowP.std()

                d["std"].append(std)
                mean=nodeWindowP.mean()
                #if(mean<1):print(mean)
                d["mean"].append(mean)
                var=0
                if (nodeWindowP.var()>var): var=nodeWindowP.var()
                d["var"].append(var)
                d["label"].append(cases[i])
                d["hop"].append(data[i][j].hop)
                d["type"].append(casesAccuracy[i])
                d["outliers"].append(getOutliers(nodeWindow)["rtt"].count())
                missing=window-nodeWindow.count()
                d["node"].append(data[i][j].ip)
                mP=getPercentageMissingPackets(nodeWindow,window)
                PL=0
                if(mP>30):
                    PL=1
                    PL=mP
                d["packet loss"].append(PL)



    def analyze_network(directory,plots,pings,window):
    cases=[]
    casesAccuracy=[]
    for row in plots:
        cases.append(row[1])
        casesAccuracy.append(row[2])
        data=import_Cooja2(plots)
    pings=getPings(data)
    #All data collection is in variable node that is a list of list of nodes
    #3 nets input x 9 nodes by net

    d={ "label":[],
       "type":[],
        "count":[],
        "std":  [],
        "mean": [],
        "var":  [],
        "hop":[],

       "packet loss":[],
       "outliers":[],
       "node":[]
    }
    #count=[]
    labels=[]
    var=[]
    #window=100
    #stats=pd.DataFrame(columns=columns)

    for i in range(len(data)):
        #window=pings[i]

        for j in range(len(data[i])):
            #n=pings[i]
            n=100
            window=50
            #print(n)
            for z in range(0,n,int(window)):
                #if(z+window>n):break
                #print(z,z+window)

                #df1 = df1.assign(e=p.Series(np.random.randn(sLength)).values)
                node=data[i][j].pkts
                name=str(j)+" "+cases[i]
                nodeWindow=node[(node["seq"]<z+window) & (node["seq"]>=z)]
                nodeWindowP=nodeWindow["rtt"]
                d["count"].append(nodeWindowP.count())
                #Case without outliers
                #Case with outliers
                std=0
                if (nodeWindowP.std()>10):
                    std=1
                    std=nodeWindowP.std()

                d["std"].append(std)
                mean=nodeWindowP.mean()
                #if(mean<1):print(mean)
                d["mean"].append(mean)
                var=0
                if (nodeWindowP.var()>var): var=nodeWindowP.var()
                d["var"].append(var)
                d["label"].append(cases[i])
                d["hop"].append(data[i][j].hop)
                d["type"].append(casesAccuracy[i])
                d["outliers"].append(getOutliers(nodeWindow)["rtt"].count())
                missing=window-nodeWindow.count()
                d["node"].append(data[i][j].ip)
                mP=getPercentageMissingPackets(nodeWindow,window)
                PL=0
                if(mP>30):
                    PL=1
                    PL=mP
                d["packet loss"].append(PL)



    stats=pd.DataFrame(d)

    dataK=stats.drop([
        "label",
        "mean",
        "var",
        "std",
        #"packet loss",
        "outliers",
        "hop",
        "count",
        "node",
        #"type"
    ],axis=1)
    dataK=dataK.fillna(0)
    correction=[]
    correction_alt=[] #fr 3 cluster 0:normal net 1:bh net 2:bh
    col=np.array(dataK["type"])
    dataK=dataK.drop(["type"],axis=1)
    #Creating simple array to correct unsupervised learning
    #NB as it is unsupervised could happen that the correction are inverted
    for i in range(len(col)):
        el=d["type"][i]
        if el=="normal":
            correction.append(1)
            correction_alt.append(0)

        else:
            #print(el=="BH2" and i==BlackHole[2])
            correction.append(0)
            correction_alt.append(1)


    dataC=stats["label"]

    #Y = data[['var']]

    #X = data[['std']]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dataK)
    labels = kmeans.predict(dataK)
    centroids = kmeans.cluster_centers_
    labels=accuracy_score_corrected(correction,labels)
    predicted=[]
    for i in range(len(labels)):

        if(labels[i]==1):
            predicted.append("normal")
        else: predicted.append("BH")

    #print(len(predicted))
    stats["predicted"]=pd.Series(np.array(predicted))
    stats["predicted number"]=pd.Series(np.array(labels))
    stats["correction number"]=pd.Series(np.array(correction))
    stats_csv=stats[[
        "label",
        "type",
        "predicted",
        "packet loss",
        "outliers",
        "std",
        "hop",
        "node",
        "mean"


          ]]
    stats_csv.to_csv("results_kmeans.csv", sep='\t', encoding='utf-8')
    stats.head()
    net_results={
       "case":[],
        "predicted":[],
        "real":[]
    }
    #print(stats["predicted number"])
    for case in range(len(cases)):
        subset=stats[stats["label"]==cases[case]]
        mean_predicted=str(subset["predicted number"].mean()*100)+"% normal"
        net_results["case"].append(cases[case])
        net_results["predicted"].append(mean_predicted)
        net_results["real"].append(casesAccuracy[case])



    results=pd.DataFrame(net_results)
    stats_csv.to_csv("results_network_kmeans.csv", sep='\t', encoding='utf-8')
