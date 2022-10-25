source('./payoff_function.r')
library(lubridate)
library(MASS)

Visualize<-function(S){
  # endindex=ncol(S)
  minS=min(S);maxS=max(S) # the y-limits of the plot
  noS<-nrow(S)
  cl<-rainbow(noS) # vector of rainbow colors
  plot(S[1,],type="l",ylim=c(minS,maxS),col=cl[1])
  if(noS>1){
    for(i in 2:noS){
      lines(S[i,],col=cl[i])
    }
  }
}


SimMultiGBMexact<-function(S0,v,Sigma,Deltat,T){ #simulate three assets' paths (once)
  m=T/Deltat # number of periods
  p=length(S0)
  S=matrix(0,nrow=p,ncol=m+1)
  S[,1]=S0
  Z<-mvrnorm(m,v*Deltat,Sigma*Deltat)
  for(j in 2:(m+1)){
    S[,j]=exp(log(S[,j-1])+Z[j-1,])
  }
  S
}


extract_data<-function(path, start_date, end_date){
  data = read.csv(path)
  data$Date = as.Date(strptime(data$Date, '%m/%d/%Y'))
  data$Close.Last = as.double(substring(data$Close.Last, 2))
  data = data[c('Date', 'Close.Last')]
  data = data[data$Date>= as.Date(start_date),]
  data = data[data$Date<=as.Date(end_date),]
  
  return (data)
}


calculate_ln_returns <- function(data){
  return (diff(log(data[order(data$Date), 'Close.Last']), lag=1))
}

calculate_sigma_hat <- function(ln_return){
  n = length(ln_return)
  return (sqrt((1/(n-1)) * sum((ln_return-mean(ln_return))^2)))
}

w_1 = 1/3; w_2 =1/3; w_3=1/3;
start_date = as.Date('2022-08-19')
end_date = as.Date('2022-10-24')
maturity_date = as.Date('2023-08-22')
q2 = as.Date('2022-08-19')+months(6)
q2_to_maturity = sum(!weekdays(seq(q2, maturity_date, "days")) %in% c("Saturday", "Sunday"))
q3 = as.Date('2022-08-19')+months(9)
q3_to_maturity = sum(!weekdays(seq(q3, maturity_date, "days")) %in% c("Saturday", "Sunday"))

cur_date = start_date
num_weekdays = sum(!weekdays(seq(start_date, maturity_date, "days")) %in% c("Saturday", "Sunday"))
total_days = num_weekdays
predicted_option_price = c()
expected_payoff_at_maturity = c()

while(cur_date<=end_date){
  print(cur_date)
  if (wday(cur_date)==7 | wday(cur_date)==1){
    cur_date = cur_date + days(1)
    next
  }
  
  historical_start = cur_date - years(1)
  googl = extract_data('./data/24-10-2022/googl.csv', historical_start, cur_date)
  aapl = extract_data('./data/24-10-2022/aapl.csv', historical_start, cur_date)
  amzn = extract_data('./data/24-10-2022/amzn.csv', historical_start, cur_date)
  
  if (!(any(amzn$Date==as.Date(cur_date)))){
    cur_date = cur_date + days(1)
    num_weekdays = num_weekdays - 1
    next
  }
  
  googl_ln_returns = calculate_ln_returns(googl)
  aapl_ln_returns = calculate_ln_returns(aapl)
  amzn_ln_returns = calculate_ln_returns(amzn)
  ln_returns = cbind(googl_ln_returns,aapl_ln_returns, amzn_ln_returns)
  
  colnames(googl) = c('Date', 'GOOGL')
  colnames(aapl) = c('Date', 'AAPL')
  colnames(amzn) = c('Date', 'AMZN')
  stock_prices = merge(googl, aapl, by='Date')
  stock_prices = merge(stock_prices, amzn, by='Date')
  
  stock_prices = as.matrix(stock_prices)
  stock_prices<-cbind(as.numeric(as.vector(stock_prices[,2])),as.numeric(as.vector(stock_prices[,3])),as.numeric(as.vector(stock_prices[,4])))
  no = nrow(stock_prices)
  
  
  v = apply(ln_returns, 2, mean)
  sigma = cov(ln_returns)
  
  Nsim=1000
  T=num_weekdays
  dt=1
  m=T/dt
  n0 = nrow(stock_prices)
  S0=stock_prices[n0,]
  S1=matrix(0,Nsim,m+1)
  S2=matrix(0,Nsim,m+1)
  S3=matrix(0,Nsim,m+1)
  set.seed(4518)
  for(i in 1:Nsim){
    S<-SimMultiGBMexact(S0,v,sigma,dt,T)
    S1[i,]=S[1,]
    S2[i,]=S[2,]
    S3[i,]=S[3,]
  }
  
  # TODO: Change the payoff quarterly function?
  payoff_at_maturity = c()
  payoff_at_q2 = c()
  payoff_at_q3 = c()
  for (i in 1:Nsim){
      # payoff = append(payoff, payoff_maturity(aapl=S2[i,], amzn=S3[i,], googl=S1[i,]))
      payoff_at_maturity = append(payoff_at_maturity, payoff_maturity(aapl=S2[i,], amzn=S3[i,], googl=S1[i,]))
      if (cur_date<q2){
        payoff_at_q2 = append(payoff_at_q2, payoff_quarter(S2[i,][1:(length(S2[i,])-q2_to_maturity)], S3[i,][1:(length(S3[i,])-q2_to_maturity)], S1[i,][1:(length(S1[i,])-q2_to_maturity)], 2))
      }
      
      if (cur_date<q3){
        payoff_at_q3 = append(payoff_at_q3, payoff_quarter(S2[i,][1:(length(S2[i,])-q3_to_maturity)], S3[i,][1:(length(S3[i,])-q3_to_maturity)], S1[i,][1:(length(S1[i,])-q3_to_maturity)], 3))
      }
  }
    
  cur_expected_payoff = mean(payoff_at_maturity)
  expected_payoff_at_maturity = append(expected_payoff_at_maturity, cur_expected_payoff)
  r= 0.04716
  option_price = exp(-r*1)*cur_expected_payoff*w_1 + exp(-r/2)*mean(payoff_at_q2)*w_2 + exp(-r*3/4)*mean(payoff_at_q3)*w_3
  predicted_option_price = append(predicted_option_price, option_price)
  
  cur_date = cur_date + days(1)
  num_weekdays = num_weekdays-1
}


predicted_option_price
dev.new(width = 20, height = 5, unit = "cm")
plot(predicted_option_price, type='l')
expected_payoff_at_maturity


