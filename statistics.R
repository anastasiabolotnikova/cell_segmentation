data_estim <- read.csv("results.txt", header = F, stringsAsFactors=FALSE)
plot(data_estim$V3[1:200], type = "l", ylim = c(0,100), xlab = "Photos", ylab="Scores", lwd=3, col="blue")
data_given <- read.csv("info_scorings_clean.csv")
data_given$score <- (data_given$X + data_given$X.1)/2

Scores <- c()
Up <- c()
Low <- c()
Diffs <- c()

hit <- 0
i <- 1
for (i in c(1:500)){
  el <- data_estim[i, ]
  lower <- data_given[data_given$file_name==el$V1,]$X
  upper <- data_given[data_given$file_name==el$V1,]$X.1
  points(i, lower, col="green")
  if(el$V2<upper && el$V2>lower){
    hit <- hit + 1
  }else{
    Diffs <- c(Diffs, abs(el$V2-(lower+upper)/2))
  }
  Up <- c(Up, upper)
  Low <- c(Low, lower)
  Scores <- c(Scores, (lower+upper)/2)
  i <- i+1
}

points(Up, col="red", type="l")
points(Low, col="green", type="l")
#points(Scores, col="green", type="l")

# How many out of all fall into the bounds
hit <- hit/500

# Correlations
cor(Scores, data_estim$V2)
cor(Scores, data_estim$V3)

# Difference in prediction (case of incorrect prediction)
boxplot(Diffs, ylab="Error")
