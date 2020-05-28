install.packages("missForest")
library(missForest)
install.packages("corrplot")
library(corrplot)
#installed libraries for corelation testing
data<-read.csv("wine.csv")
#data readed
data<-data[,-1]#instance number of data cleaned
data
cor<-cor(data)
data
cor #checked corelation
corrplot(cor,"circle","lower")
corrplot(cor,"number","lower")
install.packages("devtools")
library(devtools)
install_github("vgv/ggbiplot")
PCA.data<-prcomp(data[,c(1:13)],center=TRUE,scale. = TRUE)
#PCA applied and checekd
PCA.data
names(data) <- NULL
write.csv(data,'data.csv')