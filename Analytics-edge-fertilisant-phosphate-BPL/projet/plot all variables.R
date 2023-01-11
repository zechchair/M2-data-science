data_frame=train
coll=1
for (coll in seq(1,ncol(data_frame),1)){
  
  n=toString(colnames(data_frame)[coll])
  jpeg(paste(n, "jpeg", sep="."))
  plot(x = data_frame[,coll], y= data_frame$BPL_B,type='p',xlab = n,main = toString(zero_perc[coll]))
  dev.off()
}
