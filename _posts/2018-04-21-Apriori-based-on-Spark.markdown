---
layout: post
title:  "apriori algorithm"
date:   2018-04-21
excerpt: "How to implement the apriori algorithm?"
image: "/images/apriori1.jpg"
---

## Introduce Apriori
The Apriori algorithm was proposed by Agrawal and Srikant in 1994. Apriori is designed to operate on databases containing transactions (for example, collections of items bought by customers, or details of a website frequentation). Other algorithms are designed for finding association rules in data having no transactions (Winepi and Minepi), or having no timestamps (DNA sequencing). Each transaction is seen as a set of items (an itemset).  ———from wikipedia

Recursive method is chosen in Apriori to get a result.

1.Firstly,Selecting one item to calculate the rate and there is parameter X,which means if the rate of each item in the whole sample is lower than X,it would be dropped and the others would be put in next round.

2.Then items are regarded as sets,expanding sets by put other item in it and calculate them again by same method,continue the loop.

3.If we will get empty set at next round in loop,end the loop and the current set is out final result.
 
Here are a graph to show the algorithm in detail.(parameter:50%)

![image](/images/explain_apriori.png)

## Significant code

```
// the first round in loop,the parameter means origin data set,And the Arraylist is a collector of item.
public String get_Maxfrequent(HashMap<String,ArrayList<String>> data_set, Double min, int size)
    {

        Map<String,Integer> result_record = new HashMap<String, Integer>();
        ArrayList<ArrayList<String>> para = new ArrayList<ArrayList<String>>();
        for(Map.Entry<String,ArrayList<String>> temp:data_set.entrySet())
        {
            ArrayList<String> temp_value=temp.getValue();
            for(String index:temp_value)
            {
                if(result_record.containsKey(index))
                {
                    Integer old_value=result_record.get(index);
                    result_record.put(index,old_value+1);
                }
                else
                {
                    result_record.put(index,1);
                }
            }
        }
        for(Map.Entry<String,Integer> temp:result_record.entrySet())
        {
            double d_value = (double) temp.getValue().intValue();
            if(min<=(d_value/size))
            {
                ArrayList<String> node_1 = new ArrayList<String>();
                node_1.add(temp.getKey());
                para.add(node_1);
                type_sumer.add(temp.getKey());
            }
        }
        return get_Maxfrequent(data_set,para,min,size);
    }

    public String get_Maxfrequent(HashMap<String,ArrayList<String>> data_set,ArrayList<ArrayList<String>> parameter,Double min,int size)
    {
        Map<Integer,Integer> accumlated_result = new HashMap<Integer, Integer>();
        ArrayList<ArrayList<String>> para = new ArrayList<ArrayList<String>>();
        ArrayList<ArrayList<String>> new_list=get_newStoreList(parameter);// expending each item set in this function
        for(Map.Entry<String,ArrayList<String>> temp:data_set.entrySet())
        {
            for(ArrayList<String> store_temp:new_list)
            {
                ArrayList<String> temp_value=temp.getValue();
                if(find_value(store_temp,temp_value))
                {
                    Integer inert_index= new_list.indexOf(store_temp);
                    if(accumlated_result.containsKey(inert_index))
                    {
                        Integer old_value=accumlated_result.get(inert_index);
                        accumlated_result.put(inert_index,old_value+1);
                    }
                    else {
                        accumlated_result.put(inert_index,1);
                    }
                }
            }
        }
        for(Map.Entry<Integer,Integer> temp_record:accumlated_result.entrySet())
        {
            double d_value = (double) temp_record.getValue().intValue();
            if(min<=(d_value/size))
            {
                ArrayList<String> iter = Lists.newArrayList(new_list.get(temp_record.getKey()));
                para.add(iter);
            }
        }
        if(null==para || 0==para.size())
        {
            StringBuilder SB = new StringBuilder();
            for(ArrayList<String> temp:parameter)
            {
                SB.append(temp.toString());
                SB.append(".");
            }
            return SB.toString();
        }
        else
            return get_Maxfrequent(data_set,para,min,size);

    }
```

## SUMMARY
I just implement the algorithm by recursive function in Java.There is a copy of complete code,https://github.com/zkeal/SPARK_-MachineLearning/tree/master/src/src/Apriori

By the way,all my code is based on Spark,but the significant part could be used in everywhere.

Welcome any advice or BUG!
