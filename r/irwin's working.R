source('./payoff_function.r')
install.packages("lubridate")
library(lubridate)

calculate_ln_returns <- function(data){
  return (diff(log(data[order(data$Date), 'Close.Last']), lag=1))
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

# Using single variate GBM to seperately simulate the prices of the 3 stocks
r= 0.04716 # Obtained from http://www.worldgovernmentbonds.com/country/united-states/ as of 25/10/2022
start_date = as.Date('2022-08-19')
end_date = as.Date('2022-10-23')
maturity_date = as.Date('2023-08-22')
num_sim = 10000
cur_date = start_date
historical_start = cur_date-years(1)

num_weekdays = sum(!weekdays(seq(start_date, maturity_date, "days")) %in% c("Saturday", "Sunday"))

predicted_option_price = c()
expected_payoff = c()
dt = 1/252

googl = extract_data('./data/24-10-2022/googl.csv', historical_start, cur_date)
amzn = extract_data('./data/24-10-2022/amzn.csv', historical_start, cur_date)
aapl = extract_data('./data/24-10-2022/aapl.csv', historical_start, cur_date)
googl = googl[order(googl$Date), ]
amzn = amzn[order(amzn$Date), ]
aapl = aapl[order(aapl$Date), ]

AGA_prices = data.frame(aapl$Close.Last, googl$Close.Last, amzn$Close.Last)
AGA_log_prices = log(AGA_prices)
n0 = nrow(AGA_prices)
AGA_log_returns = AGA_log_prices[2:n0,]-AGA_log_prices[1:(n0-1),]
#v=apply(AGA_log_returns,2,mean)/dt
# risk neutral world
v = c(r, r, r)
Sigma=cov(AGA_log_returns)/dt


library(MASS)
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

# S1: apple, S2: google, S3: amazon
Nsim=10;T=1;dt=1/252
m=T/dt
S0=AGA_prices[n0, ]
S0 = as.numeric(S0)
S1=matrix(0,Nsim,m+1)
S2=matrix(0,Nsim,m+1)
S3=matrix(0,Nsim,m+1)
set.seed(4518)
for(i in 1:Nsim){
  S<-SimMultiGBMexact(S0,v,Sigma,dt,T)
  S1[i,]=S[1,]
  S2[i,]=S[2,]
  S3[i,]=S[3,]
}

HistS1<-matrix(rep(AGA_prices[,1],Nsim),ncol=n0,byrow=T)
wholeS1<-cbind(HistS1,S1)
Visualize(wholeS1)
