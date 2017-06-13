

#biocLite("Biostrings")

#annotation file
#chr_pos[,7] = gsub(";Parent=","",chr_pos[,7]) 
#gff3 annotation file
chr_pos = read.table("Creinhardtii_281_v5.5.gene.gff3", sep = "\t", header =T)
chr_pos = chr_pos[chr_pos[,3] == 'five_prime_UTR',] 
chr_pos = chr_pos[,c(1,4,5,7,9)]
chr_pos[,6] = nchar(as.matrix(chr_pos[,5]))
chr_pos[,7] = gsub("ID=","",chr_pos[,5]) 
chr_pos[,6] = nchar(as.matrix(chr_pos[,7])) 
lengths_list = unique(chr_pos[,6]) 
for(i in lengths_list){
    
  chr_pos[chr_pos[,6] == i,7] = substring(chr_pos[chr_pos[,6] == i,7],1,(i/2)-2)
  
}

chr_pos[,5] = chr_pos[,7] 
chr_pos = chr_pos[,c(1,2,3,4,5)] 
chr_pos[,1] = gsub("chromosome_","",chr_pos[,1])  
chr_pos[,1] = gsub("scaffold_","",chr_pos[,1]) 
rownames(chr_pos) = chr_pos[,5]

#fasta file

fileName="creinhardtii_281_v5.0.fa"
raw=readChar(fileName, file.info(fileName)$size)
chromosome_start = unlist(gregexpr(pattern ='>',raw))
chromosome_table = matrix(0,length(chromosome_start),2)
### still fine at this point
for(i in 1:9){
  
  start = chromosome_start[i] + nchar('>chromosome_1\n')
  end = chromosome_start[(i +1)] - 2
  chromosome_table[i,1] = i
  chromosome_table[i,2] = substring(raw,start,end)  

}

for(i in 10:17){
  
  start = chromosome_start[i] + nchar('>chromosome_10\n')
  end = chromosome_start[(i +1)] - 2
  chromosome_table[i,1] = i
  chromosome_table[i,2] = substring(raw,start,end)
}

for(i in 18:53){ ###why was this 52 earlier????
  start = chromosome_start[i] + nchar('>scaffold_10\n')
  end = chromosome_start[(i +1)] - 2
  chromosome_table[i,1] = i
  chromosome_table[i,2] = substring(raw,start,end) 
}

start = chromosome_start[54] + nchar('>scaffold_10\n')
end = chromosome_start[(54)] + 2289
chromosome_table[54,1] = 54
chromosome_table[54,2] = substring(raw,start,end) 


chromosome_table[,2] = gsub("\n","",chromosome_table[,2]) 
rownames(chromosome_table) = chromosome_table[,1]  

#grabbing sequences from the TSS

UTR.chlamy <- matrix(0,nrow(chr_pos),2)
for(i in 1:nrow(chr_pos)){#nrow(chr_pos)){  #change to 1:nrow(chr_pos)
  
  start = as.numeric(as.character(chr_pos[i,"start"])) 
  end = as.numeric(as.character(chr_pos[i,"stop"])) 
  chromosome =  chr_pos[i,"chromosome"]   
  strand = as.character(chr_pos[i,"strand"]) 
  name = chr_pos[i,"locus"] 
	if(strand == "+"){
	  UTR = substring(chromosome_table[chromosome,2],(start - 390),(start-50))   #change start+1000 to start +1????
    UTR.chlamy[i,1] = name
    UTR.chlamy[i,2] = UTR
  }
  
  if(strand == "-"){
    UTR = substring(chromosome_table[chromosome,2],(end - 500),(end))
    UTR = as.character(reverse(complement(DNAString(UTR))))
    UTR.chlamy[i,1] = name
    UTR.chlamy[i,2] = UTR 
  }
  
 }

rownames(UTR.chlamy) = UTR.chlamy[,1]
colnames(UTR.chlamy)=c("locus", "UTR sequence")
#outfile
write.table(UTR.chlamy,"UTR_chlamy.txt",
            sep = "\n",eol = "\n", row.names = F, col.names = T)



