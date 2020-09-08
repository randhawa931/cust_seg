# customer_segmentation  
Datebase downloaded from UCI repository.It consists of 8 attributes(customer_id,invoice no, quantity, stock code, invoice_date etc) and more than 500K rows.  
RFM technique  
Recency= the most recent day that customer made a purchase.Less the number of days, higer the value of Recency score.  
Frequency= number of transactions made by the customer.  
Monetary=the  amount of money spent by the customer.  

Once recency, frequency and monetary is calculated R_score, F_score & M_score in calculated. 
Each is ranked between 1-5.  
Depending on these scores the customers are segmented in various categories:  
  hibernating  
  at risk  
  can't loose   
  loyal customer  
  about to sleep  
  promising  
  new customers  
  champion  
Used kmean clustering to plot snake graphs and heat maps. 

