#----------Initial Settings and Pre Processing ------------
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("graph")

library(bnlearn) 
library(gRain) 
library(graph)
library(parallel)
library(bnclassify)
library(tidyr)
library(naniar)
library(dplyr)

source('functions.R')
bd <- read.csv("C:/Users/Serena/Desktop/MACHINE LEARNING/assigment 2/asdm-2019/breast-cancer-wisconsin.data", header=FALSE)
View(bd)
dim(bd)

#rename columns
colnames(bd) <- c("id","Clump_Thickness","Uniformity_Cell_Size","Uniformity_CellShape", "Marginal_Adhesion","Single_Epithelial_Cell_Size", "Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class")    
#remove the first column id 
bd$id<-NULL
summary(bd)
#check object type
str(bd)

#bnlearn doens't accept int variable, change the variables type in factors
bd$Clump_Thickness<-as.factor(bd$Clump_Thickness)
bd$Uniformity_Cell_Size<-as.factor(bd$Uniformity_Cell_Size)
bd$Uniformity_CellShape<-as.factor(bd$Uniformity_CellShape)
bd$Marginal_Adhesion<-as.factor(bd$Marginal_Adhesion)
bd$Single_Epithelial_Cell_Size<-as.factor(bd$Single_Epithelial_Cell_Size)
bd$Bare_Nuclei<-as.factor(bd$Bare_Nuclei)
bd$Bland_Chromatin<-as.factor(bd$Bland_Chromatin)
bd$Normal_Nucleoli<-as.factor(bd$Normal_Nucleoli)
bd$Mitoses<-as.factor(bd$Mitoses)
bd$Class<-as.factor(bd$Class)
str(bd)

#missing value here are denoted with ?, replace the ? with na values and drop them
bd <- bd %>% dplyr::na_if("?")
sum(is.na(bd))
bd <- drop_na(bd)
anyNA(bd)
dim(bd)

#class distribution 
bd %>%
  group_by(Class)  %>%
  count()
table(bd$Class)/nrow(bd)

#Splitting the dataset
set.seed(100)
index_1 <- sample(1:nrow(bd), round(nrow(bd) * 0.8))
train <- bd[index_1, ]
test  <- bd[-index_1, ]
dim(train)
dim(test)

#class distribution in train 
train %>%
  group_by(Class)  %>%
  count()
table(train$Class)/nrow(train)
is.na(train)

#-----------Learning structure----------

#Based on score + search
#Hill Climbing network structures 
hc.breast.bic <- hc(train, score = "bic")
hc.breast.aic<- hc(train, score = "aic")
hc.breast.loglik<- hc(train, score = "loglik")
hc.breast.k2<- hc(train, score = "k2")

plot(hc.breast.bic)
plot(hc.breast.aic)
plot(hc.breast.loglik) #most complex
plot(hc.breast.k2)

#Tabu network structures
tabu.breast.bic <- tabu(train, score = "bic")
tabu.breast.aic<- tabu(train, score = "aic")
tabu.breast.loglik<- tabu(train, score = "loglik")
tabu.breast.k2<- tabu(train, score = "k2")

plot(tabu.breast.bic)
plot(tabu.breast.aic)
plot(tabu.breast.loglik) #most complex
plot(tabu.breast.k2)

score(hc.breast.bic, train, type="bic")
score(hc.breast.bic, train, type="aic")
score(hc.breast.bic, train, type="loglik")
score(hc.breast.bic, train, type="k2")

#Which node contributes most to the BIC score?
score(hc.breast.bic, train, type="bic", debug = TRUE)
#¿Cuál? depende del loglik, mayor o menor o del penalty

#How many free parameters in the network?
nparams(hc.breast.bic, train)
nparams(hc.breast.aic, train)
nparams(hc.breast.loglik, train)
nparams(hc.breast.k2, train)

#Find out the penalization coefficient for BIC by print out hc.breast
hc.breast.bic 
hc.breast.aic
hc.breast.loglik
hc.breast.k2

#compare
compare(hc.breast.aic, hc.breast.bic, arcs =TRUE)
#bic network as reference
unlist(compare(hc.breast.bic, hc.breast.aic))
graphviz.compare(hc.breast.aic, hc.breast.bic)
unlist(compare(hc.breast.bic, hc.breast.loglik))
unlist(compare(hc.breast.bic, hc.breast.k2))

#aic network as reference
unlist(compare(hc.breast.aic, hc.breast.loglik))
unlist(compare(hc.breast.aic, hc.breast.k2))
unlist(compare(hc.breast.aic, hc.breast.bic))

#loglik as reference
unlist(compare(hc.breast.loglik, hc.breast.k2))
unlist(compare(hc.breast.loglik, hc.breast.aic))
unlist(compare(hc.breast.loglik, hc.breast.bic))

#k2 as reference 
unlist(compare(hc.breast.k2, hc.breast.loglik))
unlist(compare(hc.breast.k2, hc.breast.aic))
unlist(compare(hc.breast.k2, hc.breast.bic))


#Constraint-based structure learning
gs.breast <- gs(x = train)
gs.breast <- cextend(gs.breast)
gs.breast
iamb.breast <- iamb(x = train)

score(gs.breast, train, type = "bic")
score(gs.breast, train, type = "aic")
score(gs.breast, train, type = "loglik")

plot(gs.breast)
plot(iamb.breast)
unlist(compare(gs.breast, hc.breast.loglik))

graphviz.compare(cpdag(hc.breast.bic), cpdag(gs.breast))
shd(gs.breast, hc.breast.bic)
#comparison k-fold cross validation 
bn.cv(bd, 'hc', loss = "logl")
bn.cv(bd, 'tabu', loss = "logl")
bn.cv(bd, 'gs', loss = "logl")
bn.cv(bd, 'iamb', loss = "logl")

# the chosen network is hc.breast.bic 

#---------Learning Parameter----------

#maximum likelihood
breast.bn.fit <- bn.fit(hc.breast.bic, train)
breast.bn.fit
bn.fit.barchart(breast.bn.fit$Uniformity_CellShape)
breast.bn.fit$Uniformity_CellShape$prob

#bayesian estimation
breast.bpe <- bn.fit(hc.breast.bic, train, method = "bayes", iss = 1)
breast.bpe
breast.bpe$Uniformity_CellShape$prob
bn.fit.barchart(breast.bpe$Uniformity_CellShape)

#------------INFERENCE----------------

#APPROXIMATED
#Forward Sampling Draw 15 samples

set.seed(0)
samples.asia <- cpdist(breast.bn.fit, nodes = nodes(breast.bn.fit),
                       evidence = TRUE, n = 15)
summary(samples.asia)

t <- table(samples.asia[, c('Uniformity_Cell_Size', 'Class', 'Uniformity_CellShape')])
prop.table(t)

#Direct probability
set.seed(0)
ep <- cpquery(breast.bpe, event = (Uniformity_CellShape == "9" & Uniformity_Cell_Size == "7" & Class == "4"), evidence = TRUE)
ep

#EXACT
gr.asia <- as.grain(breast.bpe)
q <- querygrain(gr.asia,
                nodes = c("Class", "Uniformity_Cell_Size","Uniformity_CellShape"), type = "joint")
q


