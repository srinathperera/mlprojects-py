#install.packages("forecast")
library("forecast")

plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

printAccuracy <- function(y_actual, y_predicted, percentage_cutoof)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}


forecastWithAutoArima <- function(x_all, yearfreq, startyear, startfreqpoint)
{
  data_size <- length(x_all)
  train_size <- floor(data_size*0.7)
  x_train <- x_all[0:train_size]
  x_test <- x_all[train_size:data_size]
  
  #start value is the year and frequency within that e.g. c(2006,503604) means 503604 in 2006 year. and each year has
  #525600 instances
  powerTs <- ts(x_train, frequency=yearfreq, start=c(startyear,startfreqpoint))
  train_end = end(powerTs)
  powerTestTs <- ts(x_test, frequency=yearfreq, start=c(train_end[1], train_end[2]+1))
  #plot.ts(powerTs)
  
  #forecast
  arimaModel <- auto.arima(powerTs)
  #forecast_time_hours <- ceiling(length(x_test)/60)
  #print(cat("forecast_time_hours=", forecast_time_hours))
  powerforecast <- forecast.Arima(arimaModel, h=length(x_test))
  accuracy(powerforecast)
}


#powerData <- read.csv("/Users/srinath/playground/data-science/keras-theano/TimeSeriesRegression/household_power_consumption1000.txt", sep = ";")
powerData <- read.csv("/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/data/household_power_consumption200k.txt", sep = ";")
forecastWithAutoArima(powerData$Global_active_power, 525600, 2006,503604)

data <- read.csv("/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/data/milk_production.csv", sep = ",")
forecastWithAutoArima(data$Production, 12, 1962,0)

data <- read.csv("/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/data/bikesharing_hourly.csv", sep = ",")
forecastWithAutoArima(data$cnt, 13560, 2011,0)

data <- read.csv("/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/data/USDvsEUExchangeRateFixed.csv", sep = ",")
forecastWithAutoArima(data$ExchangeRate, 365, 1999,4)


data <- read.csv("/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/data/applestocksfixed.csv", sep = ",")
forecastWithAutoArima(data$Close, 365, 1980,345)


#http://finance.yahoo.com/q/hp?s=AAPL&a=11&b=12&c=1980&d=05&e=1&f=2016&g=d
appleSotcksDf = pd.read_csv('./data/applestocksfixed.csv')
check_assending(appleSotcksDf, 'Date','%Y-%m-%d')
appleSotcks = appleSotcksDf['Close']

#data_size <- length(x_all)
#train_size <- floor(data_size*0.7)
#x_train <- x_all[0:train_size]
#x_test <- x_all[train_size:data_size]

#start value is the year and frequency within that e.g. c(2006,503604) means 503604 in 2006 year. and each year has
#525600 instances
#powerTs <- ts(x_train, frequency=525600, start=c(2006,503604))

#train_end = end(powerTs)

#powerTestTs <- ts(x_test, frequency=525600, start=c(train_end[1], train_end[2]+1))
#plot.ts(powerTs)



#arimaModel <- auto.arima(powerTs)

#forecast_time_hours <- ceiling(length(x_test)/60)
#print(cat("forecast_time_hours=", forecast_time_hours))
#forecast_time_hours
#powerforecast <- forecast.Arima(arimaModel, h=length(x_test))
#accuracy(powerforecast)

# following did not work. Second argument provide test data
#accuracy(powerforecast, powerTestTs)

#arimaModel



#revenuearimaforecastdf <- as.data.frame(revenuearimaforecast)
#forcastedValues<- revenuearimaforecastdf$"Point Forecast"
#c(sum(forcastedValues[1:12]), sum(forcastedValues[12:24]), sum(forcastedValues[24:36]))



