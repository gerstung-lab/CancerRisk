rm(list=ls())

model_comparison <- read.csv("~/Desktop/ee/ukb_extract/main/tables/model_comparison.csv")

p <- as.numeric(unlist(unname(model_comparison[, c(8, 9, 10, 11, 12, 13)])))

write.csv(matrix(p.adjust(p, method ='hommel', n = length(p)), ncol=6), '/Users/alexwjung/Desktop/p_adj_ukb.csv')


model_comparison <- read.csv("~/Desktop/ee/danish_extract/main/tables/model_comparison.csv")

names(model_comparison)[12]

p <- as.numeric(unlist(unname(model_comparison[, c(7, 8, 9, 10, 11)])))

write.csv(matrix(p.adjust(p, method ='hommel', n = length(p)), ncol=5), '/Users/alexwjung/Desktop/p_adj_denmark.csv')

