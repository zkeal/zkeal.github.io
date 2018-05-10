---
layout: post
title:  "SVM algorithm"
date:   2018-04-27
excerpt: "How to implement the SVM algorithm?"
image: "/images/svm.jpg"
---

## Introduce SVM
Support Vector Machine in the field of machine learning is a supervised learning model, which is usually used for pattern recognition, classification and regression analysis.It implements classification by building a hyperplane.
It uses training data to get parameters which are calculated by Lagrange multiplier method.

__Principia mathematica__ 

_AIM:_
<br>
What is out target is to search a hyperplane that has longest distance between two different type of points.Obviously we need construct an equation :g(x) = wx + b.<br>And distance of pointer to line g(x)=0 : |g(xi)|/||w||,thus we need to get the min value of ||w|| to get the max distance.

Now,this problem become to solve the equation,The minimum value of w squared. <br>

_subject to :yi[(w·xi)+b]-1≥0 (i=1,2,…,l) （l is the size of training data)_

At next we use Lagrange multiplier method to solve it.

![image](/images/lagrange.png)

So we solve for the duality of this problem, and then we take the partial derivative.

![image](/images/lagrange_01.png)

and got:

![image](/images/lagrange_02.png)

According to KKT conditions, the maximum value is solved,and calculate it by:

![image](/images/lagrange_03.png)

__NB__:
there is another method to solve the last problem ,SMO,that has been implemented by java in my github.it uses KKT and dynamic programming to solve it more quickly. 

__Significant code__

It's a part to implement the method that is used for solving this:
 
![image](/images/lagrange_02.png)

And as the result of ∑ aiy(i) = 0,we can got the key equation:

![image](/images/lagrange_04.png)

Obviously,dynamic programming to solve it,and this graph is best explain to introduce "SMO".

![image](/images/lagrange_04.png)

And solve it according the Karush-Kuhn-Tucker(KKT) condition.

A description of some parameters：　
aerph -> α ,the first parameter of lagrange.
label -> b ,the value b in wx+b.
traindata,the date set of training.tolerance,Upper limit of error.maxcounter,the max amount of loop.

```
  
   public Equations SMO(Matrix aerph, Matrix label, Matrix traindata, double tolerance, int maxcounter) throws HiveException
    {
        double C=tolerance;
        double B=0;
        int Dimension = traindata.getColumnDimension();
        int row = traindata.getRowDimension();
        try {
            int iter=0;
            while (iter < maxcounter)
            {
                for(int i=0;i<row;i++)
                {
                    double Ei = getEi(aerph,label,traindata,i,Dimension,B);
                    if((label.get(i,0)*Ei<-0.001 && aerph.get(i,0)<C)||(label.get(i,0)*Ei>0.001 && aerph.get(i,0)>0))
                    {
                        // random choice
                        int j = getRandom_index(i,row);
                        //int j=get_heuristic(aerph,label,traindata,Dimension,B,traindata.getRowDimension(),Ei);
                        double Ej = getEi(aerph,label,traindata,j,Dimension,B);
                        double Lbottom = 0;
                        double Hup =0;
                        if(label.get(i,0)!=label.get(j,0))
                        {
                            Lbottom = aerph.get(j,0)-aerph.get(i,0);
                            Lbottom = Lbottom>0?Lbottom:0;
                            Hup = aerph.get(j,0)-aerph.get(i,0)+C;
                            Hup = Hup<C?Hup:C;
                        }else {
                            Lbottom = aerph.get(j,0)+aerph.get(i,0)-C;
                            Lbottom=Lbottom>0?Lbottom:0;
                            Hup = aerph.get(j,0)+aerph.get(i,0);
                            Hup = Hup<C?Hup:C;
                        }
                        if(Lbottom==Hup)
                        {
                            break;
                        }else {
                            Matrix data_i = traindata.getMatrix(i,i,0,Dimension-1);
                            Matrix data_j = traindata.getMatrix(j,j,0,Dimension-1);

                            double eta = get_inner_product(data_i.times(2),data_j)-get_inner_product(data_i,data_i)-get_inner_product(data_j,data_j);
                            double aerph_old_i = aerph.get(i,0);
                            double aerph_old_j = aerph.get(j,0);

                            double aerph_new_j = aerph_old_j - label.get(j,0)*(Ei-Ej)/eta;
                            aerph_new_j=aerph_new_j > Hup?Hup:aerph_new_j;
                            aerph_new_j=aerph_new_j < Lbottom?Hup:aerph_new_j;
                            if(Math.abs(aerph_new_j-aerph_old_j)<0.001)
                            {
                                continue;
                            }
                            double aerph_new_i = aerph_old_i+label.get(i,0)*label.get(j,0)*(aerph_old_j-aerph_new_j);
                            double B1 = B-Ei-label.get(i,0)*(aerph_new_i-aerph_old_i)*data_i.times(data_i.transpose()).get(0,0)-label.get(j,0)*(aerph_new_j-aerph_old_j)*data_i.times(data_j.transpose()).get(0,0);
                            double B2 = B-Ej-label.get(i,0)*(aerph_new_i-aerph_old_i)*data_i.times(data_j.transpose()).get(0,0)-label.get(j,0)*(aerph_new_j-aerph_old_j)*data_j.times(data_j.transpose()).get(0,0);
                            if(aerph_new_i>0 && aerph_new_i<C)
                            {
                                B=B1;
                            }else if(aerph_new_j>0 && aerph_new_j< C)
                            {
                                B=B2;
                            }else {
                                B=B1+B2;
                            }
                            aerph.set(i,0,aerph_new_i);
                            aerph.set(j,0,aerph_new_j);
                        }
                    }
                }
                iter++;
            }
            Equations result = new Equations(aerph,label,traindata,B);
            return result;
        }catch (Exception e)
        {
            e.printStackTrace();
            throw new HiveException("Error in SMO: "+e.getMessage());
        }
    }
```

__THANKS__:The formula about principia mathematica come from [this](http://www.cnblogs.com/jerrylead).

