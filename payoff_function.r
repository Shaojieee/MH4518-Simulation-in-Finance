payoff <- function(aapl, amzn, googl){
  aapl_barrier = 85.760
  amzn_barrier = 69.115
  googl_barrier = 58.605
  if (Reduce('|', aapl<=aapl_barrier)|Reduce('|', amzn<=amzn_barrier)|Reduce('|', googl<=googl_barrier)){
    aapl_percent = tail(aapl,n=1)/aapl_barrier
    amzn_percent = tail(amzn, n=1)/amzn_barrier
    googl_percent = tail(googl, n=1)/googl_barrier
    if (aapl_percent<amzn_percent & aapl_percent<googl_percent){
      lowest_stock = 'aapl'
      lowest_percent = aapl_percent
      lowest_price = tail(aapl, n=1)
      conversion_ratio = 5.8302
    }else if (amzn_percent<aapl_percent & amzn_percent<googl_percent){
      lowest_stock = 'amzn'
      lowest_percent = amzn_percent
      lowest_price = tail(amzn, n=1)
      conversion_ratio = 7.2343
    }else{
      lowest_stock = 'googl'
      lowest_percent = googl_percent
      lowest_price = tail(googl, n=1)
      conversion_ratio = 8.5317
    }
    if (lowest_percent>100){
      return (1000*1.1)
    }else{
      return ((lowest_price * conversion_ratio) + (1000*0.1))
    }
  }else{
    return (1000*1.1)
  }
  
}
