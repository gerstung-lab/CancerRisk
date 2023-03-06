
library(survival)

args <- commandArgs()
var1 <- args[8]
print(var1)
dsplit <- "train"
dir = "/users/projects/cancer_risk/"
events=c("oesophagus", "stomach", "colorectal", "liver", "pancreas", "lung", "melanoma", "breast", "cervix_uteri", "corpus_uteri", "ovary", "prostate", "testis", "kidney", "bladder", "brain", "thyroid", "non_hodgkin_lymphoma", "multiple_myeloma", "AML", "other", "death")

dd <- read.csv(paste(dir, "tmp/", dsplit,  "_", as.character(var1), ".csv", sep='') , sep=";")

r_coef = c()
r_se = c()

r_coef_dnpr = c()
r_se_dnpr = c()

r_coef_gene = c()
r_se_gene = c()

r_coef_bth = c()
r_se_bth = c()



# all 
m = coxph(Surv(X0, X1, X2) ~X3, data=dd)
r_coef = c(r_coef, unname(summary(m)$concordance[1]))
r_se = c(r_se, unname(summary(m)$concordance[2]))

m = coxph(Surv(X0, X1, X2) ~X4, data=dd)
r_coef_dnpr = c(r_coef_dnpr, unname(summary(m)$concordance[1]))
r_se_dnpr = c(r_se_dnpr, unname(summary(m)$concordance[2]))

m = coxph(Surv(X0, X1, X2) ~X5, data=dd)
r_coef_gene = c(r_coef_gene, unname(summary(m)$concordance[1]))
r_se_gene = c(r_se_gene, unname(summary(m)$concordance[2]))

m = coxph(Surv(X0, X1, X2) ~X6, data=dd)
r_coef_bth = c(r_coef_bth, unname(summary(m)$concordance[1]))
r_se_bth = c(r_se_bth, unname(summary(m)$concordance[2]))


# female
if(var1 %in% c(11, 12)){
  r_coef = c(r_coef, 0)
  r_se = c(r_se, 0)
  
  r_coef_dnpr = c(r_coef_dnpr, 0)
  r_se_dnpr = c(r_se_dnpr, 0)
  
  r_coef_gene = c(r_coef_gene, 0)
  r_se_gene = c(r_se_gene, 0)
  
  r_coef_bth = c(r_coef_bth, 0)
  r_se_bth = c(r_se_bth, 0)
}else{
  m = coxph(Surv(X0, X1, X2) ~X3, data=dd[dd$X7==0, ])
  r_coef = c(r_coef, unname(summary(m)$concordance[1]))
  r_se = c(r_se, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X4, data=dd[dd$X7==0, ])
  r_coef_dnpr = c(r_coef_dnpr, unname(summary(m)$concordance[1]))
  r_se_dnpr = c(r_se_dnpr, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X5, data=dd[dd$X7==0, ])
  r_coef_gene = c(r_coef_gene, unname(summary(m)$concordance[1]))
  r_se_gene = c(r_se_gene, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X6, data=dd[dd$X7==0, ])
  r_coef_bth = c(r_coef_bth, unname(summary(m)$concordance[1]))
  r_se_bth = c(r_se_bth, unname(summary(m)$concordance[2]))
}

# male
if(var1 %in% c(7, 8, 9, 10)){
  r_coef = c(r_coef, 0)
  r_se = c(r_se, 0)
  
  r_coef_dnpr = c(r_coef_dnpr, 0)
  r_se_dnpr = c(r_se_dnpr, 0)
  
  r_coef_gene = c(r_coef_gene, 0)
  r_se_gene = c(r_se_gene, 0)
  
  r_coef_bth = c(r_coef_bth, 0)
  r_se_bth = c(r_se_bth, 0)
}else{
  m = coxph(Surv(X0, X1, X2) ~X3, data=dd[dd$X7==1, ])
  r_coef = c(r_coef, unname(summary(m)$concordance[1]))
  r_se = c(r_se, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X4, data=dd[dd$X7==1, ])
  r_coef_dnpr = c(r_coef_dnpr, unname(summary(m)$concordance[1]))
  r_se_dnpr = c(r_se_dnpr, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X5, data=dd[dd$X7==1, ])
  r_coef_gene = c(r_coef_gene, unname(summary(m)$concordance[1]))
  r_se_gene = c(r_se_gene, unname(summary(m)$concordance[2]))
  
  m = coxph(Surv(X0, X1, X2) ~X6, data=dd[dd$X7==1, ])
  r_coef_bth = c(r_coef_bth, unname(summary(m)$concordance[1]))
  r_se_bth = c(r_se_bth, unname(summary(m)$concordance[2]))
}


df = data.frame("ci"=r_coef, 
           "ci_se"=r_se,
           "ci_dnpr"=r_coef_dnpr, 
           "ci_se_dnpr"=r_se_dnpr,
           "ci_gene"=r_coef_gene, 
           "ci_se_gene"=r_se_gene,
           "ci_bth"=r_coef_bth, 
           "ci_se_bth"=r_se_bth)

write.csv(df, paste(dir, "main/output/", events[as.numeric(var1)+1] ,  "/data/concordance_", dsplit, ".csv", sep=''))


