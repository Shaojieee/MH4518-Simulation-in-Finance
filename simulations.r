source('./payoff_function.r')
install.packages("lubridate")
library(lubridate)

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


SimGBMexact<-function(Nsim,S0,v,sigma,Deltat,T){
  m=T/Deltat # number of periods
  S=matrix(S0,nrow=Nsim,ncol=m+1)
  for(i in 1:Nsim){
    Z<-rnorm(m)
    for(j in 2:(m+1)){
      S[i,j]=S[i,j-1]*exp(v*Deltat+sigma*sqrt(Deltat)*Z[j-1])
    }
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



# Using single variate GBM to seperately simulate the prices of the 3 stocks
r= 0.04716 # Obtained from http://www.worldgovernmentbonds.com/country/united-states/ as of 25/10/2022
start_date = as.Date('2022-08-19')
end_date = as.Date('2022-10-23')
maturity_date = as.Date('2023-08-22')
num_sim = 10000
cur_date = start_date

num_weekdays = sum(!weekdays(seq(start_date, maturity_date, "days")) %in% c("Saturday", "Sunday"))

predicted_option_price = c()
expected_payoff = c()
while(cur_date<=end_date){
  if (wday(cur_date)==7 | wday(cur_date)==1){
    
    cur_date = cur_date + days(1)
    next
  }
  print(cur_date)
  historical_start = cur_date-years(1)
  
  googl = extract_data('./data/24-10-2022/googl.csv', historical_start, cur_date)
  googl_ln_returns = calculate_ln_returns(googl)
  v_hat = mean(googl_ln_returns)
  sigma_hat = calculate_sigma_hat(googl_ln_returns)
  if (any(googl$Date==as.Date(cur_date))){
    simulated_googl = SimGBMexact(num_sim, googl[googl$Date==as.Date(cur_date), 'Close.Last'], v_hat, sigma_hat, 1, num_weekdays)
    # Visualize(simulated_googl)
  }else{
    num_weekdays = num_weekdays - 1
    cur_date = cur_date + days(1)
    next
  }
  
  
  aapl = extract_data('./data/24-10-2022/aapl.csv', historical_start, cur_date)
  aapl_ln_returns = calculate_ln_returns(aapl)
  v_hat = mean(aapl_ln_returns)
  sigma_hat = calculate_sigma_hat(aapl_ln_returns)
  if (any(aapl$Date==as.Date(cur_date))){
    simulated_aapl = SimGBMexact(num_sim, aapl[aapl$Date==as.Date(cur_date), 'Close.Last'], v, sigma, 1, num_weekdays)
    # Visualize(simulated_aapl)
    }else{
      num_weekdays = num_weekdays - 1
      cur_date = cur_date + days(1)
      next
    }
  
  amzn = extract_data('./data/24-10-2022/amzn.csv', historical_start, cur_date)
  amzn_ln_returns = calculate_ln_returns(amzn)
  v_hat = mean(amzn_ln_returns)
  sigma_hat = calculate_sigma_hat(amzn_ln_returns)
  if (any(amzn$Date==as.Date(cur_date))){
    simulated_amzn = SimGBMexact(num_sim, amzn[amzn$Date==as.Date(cur_date), 'Close.Last'], v, sigma, 1, num_weekdays)
    # Visualize(simulated_amzn)
  }else{
    num_weekdays = num_weekdays - 1
    cur_date = cur_date + days(1)
    next
  }
  
  payoff = c()
  for (i in 1:1000){
    payoff = append(payoff, payoff_maturity(simulated_aapl[i,], simulated_amzn[i,], simulated_googl[i,]))
  }
  
  cur_expected_payoff = mean(payoff)
  expected_payoff = append(expected_payoff, cur_expected_payoff)
  r= 0.04716 # Obtained from http://www.worldgovernmentbonds.com/country/united-states/ as of 25/10/2022
  option_price = exp(-r*1)*cur_expected_payoff
  predicted_option_price = append(predicted_option_price, option_price)
  cur_date = cur_date + days(1)
  num_weekdays = num_weekdays - 1
}


predicted_option_price
dev.new(width = 20, height = 5, unit = "cm")
plot(predicted_option_price, type='l')
expected_payoff
