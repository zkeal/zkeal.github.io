---
layout: post
title:  "IsolationForest algorithm"
date:   2018-08-12
excerpt: "the best algorithm for distribute system"
image: "/images/apriori1.jpg"
---

## Introduction
Anomalies are data patterns that have different data characteristics
from normal instances. The detection of anomalies
has significant relevance and often provides critical actionable
information in various application domains. For
example, anomalies in credit card transactions could signify
fraudulent use of credit cards. An anomalous spot in an astronomy
image could indicate the discovery of a new star.
An unusual computer network traffic pattern could stand
for an unauthorised access. These applications demand
anomaly detection algorithms with high detection performance
and fast execution.

## Method summery
The proposed method, called Isolation Forest or iForest,
builds an ensemble of iTrees for a given data set, then
anomalies are those instances which have short average path
lengths on the iTrees.

Isolation Forest is an excellent algorithm for anomaly detecting.it could split potints automaticly that far away from 
aggregation group.it not only effective but also extramely suitble for distribute system due to its special computing 
method.It divide data set into many small data groups so that distribute system can build tree by each data group in the 
stage of Mapper.

##The Implementation
__Definition__ 

Isolation Tree. Let T be a node of an isolation
tree. T is either an external-node with no child, or an
internal-node with one test and exactly two daughter nodes
(Tl,Tr). A test consists of an attribute q and a split value p
such that the test q < p divides data points into Tl and Tr.
Given a sample of data X = {x1, ..., xn} of n instances
from a d-variate distribution, to build an isolation
tree (iTree), we recursively divide X by randomly selecting
an attribute q and a split value p, until either: (i) the
tree reaches a height limit, (ii) |X| = 1 or (iii) all data in
X have the same values. An iTree is a proper binary tree,
where each node in the tree has exactly zero or two daughter
nodes. Assuming all instances are distinct, each instance is
isolated to an external node when an iTree is fully grown, in
which case the number of external nodes is n and the number
of internal nodes is n − 1; the total number of nodes
of an iTrees is 2n − 1; and thus the memory requirement is
bounded and only grows linearly with n.
The task of anomaly detection is to provide a ranking
that reflects the degree of anomaly. Thus, one way to detect
anomalies is to sort data points according to their path
lengths or anomaly scores; and anomalies are points that
are ranked at the top of the list. We define path length and
anomaly score as follows.

__Ananomaly score__ 

>h(x):while the maximum possible height of iTree grows in the order of n.

>the average path length of unsuccessful search in BST as:  c(n) = 2H(n − 1) − (2(n − 1)/n)

>H(i):the harmonic number and it can be estimated by ln(i) + 0.5772156649 (Euler’s constant).

>The anomaly scores of an instance x is defined as: ![image](/images/if_airo.png)

__CODE__

* ArrayList<Double> data_tree : the series of value for the data groups

* Tree_node tree_node : the root node of ITree

* int tree_high : Upper limit of the height of the resulting tree


_Build Itree_:distribute system split data set into several groups and use each group build ITree in the Mapper stage on
different node.


```
public Tree_node random_split(ArrayList<Double> data_tree,Tree_node tree_node,int tree_high) throws HiveException
{
    try {
        sum = sum +tree_high;//convinent for get E(h(x))
        if(data_tree.size()>1)
        {
            ArrayList<Double> left_tree = new ArrayList <Double>();
            ArrayList<Double> right_tree = new ArrayList <Double>();
            double min = data_tree.get(0);
            double max = data_tree.get(data_tree.size()-1);
            if(max!=min)
            {
                double result = Math.random();
                int Int_Index = (int)(result*(data_tree.size()));
                for(Double temp_node:data_tree)
                {
                    if(temp_node<data_tree.get(Int_Index))
                    {
                        left_tree.add(temp_node);
                    }
                    else
                    {
                        right_tree.add(temp_node);
                    }
                }
                tree_node.setValue(data_tree.get(Int_Index));
                data_tree.clear();
                if(left_tree.size() == right_tree.size())
                {
                    return tree_node;
                }
                tree_node.left_tree = new Tree_node(tree_high+1);
                tree_node.right_tree = new Tree_node(tree_high+1);
                tree_node.left_tree= random_split(left_tree,tree_node.left_tree,tree_high+1);
                tree_node.right_tree = random_split(right_tree,tree_node.right_tree,tree_high+1);
            }
        }
        return tree_node;
    }catch (Exception e)
    {
        throw  new HiveException("calculate_IFtree failed");
    }
}
```

_Anomaly score_:this is a traversal method to get scores according the formula of detection.

* int max_count : the count of data in per group

* double Dividing : Used to distinguish anomalies of threshold


```
public ArrayList<Double> Isolate_value(Tree_node root,int max_count,double Dividing)
    {
        int count_now  = 0;
        double temp_sum = 0;
        ArrayList<Double> result_list = new ArrayList <Double>();
        Tree_node temp_node = root;
        Stack<Tree_node> stack = new Stack <Tree_node>();
        while (temp_node.value !=null || stack.size()!=0)
        {
            if(temp_node.value==null)
            {
                temp_node=stack.pop();
                continue;
            }
            if(temp_node.left_tree!=null)
            {
                stack.push(temp_node.left_tree);
            }
            if(temp_node.right_tree!=null)
            {
                stack.push(temp_node.right_tree);
            }
            count_now++;
            temp_sum = temp_sum+ temp_node.getValue();
            double temp_avg = temp_sum/count_now;
            double result  = Math.pow(2,-temp_avg/get_EulerConstant(count_now,max_count));
            if(result>Dividing && result < 1-Dividing)
            {
                result_list.add(temp_node.getValue());
            }
            temp_node=stack.pop();
        }
        return result_list;
```

After the caculation,the normal data will be stored and merged in the stage of reduce.those data groups are combined together
to output at the end. 

## SUMMARY
This paper proposes a fundamentally different model- based method that focuses on anomaly isolation rather than normal instance
 profiling. The concept of isolation has not been explored in the current literature and the use of 
 isola- tion is shown to be highly effective in detecting anomalies with extremely high efficiency. 
 Taking advantage of anoma- lies’ nature of ‘few and different’, iTree isolates anoma- lies closer to the root of the tree as compared to normal points.
  This unique characteristic allows iForest to build partial models (as opposed to full models in profiling) and employ only a tiny proportion of training data to build ef- fective models.
  As a result, iForest has a linear time com- plexity with a low constant and a low memory requirement which is ideal for high volume data sets.
  
 ***
 paper link:[https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf]