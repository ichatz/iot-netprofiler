
import pandas as pd

class node(object):
    ip = ""
    hop= 0
    pkts=pd.DataFrame()

    # The class "constructor" - It's actually an initializer
    def __init__(self,ip,hop,pkts):
        self.ip = ip
        self.hop=hop
        self.pkts=pkts


def createNodes(dict):
  # Input: Cooja json
  # Return: create a new Node object

  nodeList=[]
  #dfList(pd.DataFrame(dict))
  for ip in dict.keys():
      pkts=pd.DataFrame(dict[ip]['pkts'])
      #(ip,hop,min_rtt,max_rtt,pkts,responses)
      #print(dict.get(ip).get("max_rtt"))
      #findMissingPackets(dict.get(ip))
      #pkts1=dict.get(ip).get("pkts")
      #pktsList=[]
      #for p in pkts1:
          #print(p.get("rtt"))
           #make_packet(rtt,pkt,ttl)
          #rtt=p.get("rtt")
          #pkt=p.get("pkt")
          #pack=packet(rtt,pkt,ttl)
          #pktsList.append(pack)
      hop=64-(int(pkts[0:1]["ttl"]))
      pkts = pkts.drop(['ttl'], axis=1)
      pkts=pkts.rename(columns={"pkt":"seq"})
      #print(type(pkts[0:1]["ttl"]))
      #print(pkts[0:1]["ttl"])
      n= node(ip,hop,pkts)

      nodeList.append(n)
      #print(type(nodeList[0].pkts[0]))

  return nodeList