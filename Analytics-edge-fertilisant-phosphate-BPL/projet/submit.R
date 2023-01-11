t=read.csv("test_1.csv")
BPL_B=Reg.pred
OBJECTID=t$OBJECTID
subb=data.frame(OBJECTID,BPL_B)

write.csv(subb,"C:\\Users\\zakaria.echchair\\Desktop\\new proj\\newgbm2.csv", row.names = FALSE)

