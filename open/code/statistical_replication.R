# get local directory from bash 
argv <- getwd()
setwd(argv)

# Load libraries 
if (!require("pacman")) install.packages("pacman", repos = "http://cran.us.r-project.org")
pacman::p_load(MASS, stargazer,ggalluvial,ggplot2,dplyr)

library(MASS)
library(stargazer)
library(ggalluvial)
library(ggplot2)
library(dplyr)


##########
# Table 2 
##########

# read in data 
model_data <- read.csv("./data/UserStats.csv")

## Get descriptives for features 
descriptive_info <- model_data %>%
  select(issued_posts,issued_comment,issued_like,received_views)

sapply(descriptive_info, function(x) c( "Stand dev" = sd(x), 
                                        "Mean"= mean(x,na.rm=TRUE)))


## Checking for overdispersion
c(mean=mean(model_data$exp_influence),
  var=var(model_data$exp_influence),
  ratio=var(model_data$exp_influence)/mean(model_data$exp_influence))

# chi squared test 
poi_model <-  glm(exp_influence~issued_posts+issued_comment+issued_like+received_views,
                  data=model_data,family=poisson())

qchisq(0.95, df.residual(poi_model))
deviance(poi_model)

pr <- residuals(poi_model,"pearson")
sum(pr^2)


## Fit Quasi-Poisson
qp <- glm(exp_influence~issued_posts+issued_comment+issued_like+received_views,
          data=model_data,family=quasipoisson)


## Fit Gamma-Poisson
nb_model <- glm.nb(exp_influence~issued_posts+issued_comment+issued_like+received_views,
                   data=model_data,control=glm.control(maxit=50))


# Output table 
stargazer::stargazer(qp, nb_model, 
                     omit.stat = c("ser", "f","ll","theta","aic"),
                     column.labels  = c(
                       "Quasi-Poisson", "Gamma-Poisson (NB)"),
                     dep.var.labels.include = F,
                     out.header = F, header = F, 
                     colnames = F,
                     model.numbers = F,
                     model.names =F,
                     covariate.labels = c("Issued Posts","Issued Comments",
                                          "Issued Likes","Received Views "),
                     out = "./output/table2.tex"
)


##########
# Figure 4 
##########

influence <- read.csv("./data/InfluenceStats.csv")
influence$bin <- factor(influence$bin,levels = c("High","Med","Low"))


p1 <- ggplot(influence,
             aes(x = time, stratum = bin, alluvium = user,
                 fill = bin, label = bin)) +
  scale_fill_brewer("Mentions",type = "qual",
                    palette = "Set2",labels=c(">1","1","0"),
                    guide = guide_legend(reverse = TRUE) ) +
  geom_flow(stat = "alluvium", lode.guidance = "frontback",
            color = "darkgray") +
  geom_stratum() + theme_bw() +
  theme(legend.position = "bottom") + labs(x="Survey Round",
                                           title="Mentioned as Influential User")

pdf("./output/alluvial.pdf",height = 5,width=9)
p1
dev.off()

