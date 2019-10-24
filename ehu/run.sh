oFile=kk.org
embDir=../../wembeddings
if [ "$#" -gt "0" ]
then
	oFile=$1
fi

if [ "$#" -gt "1" ]
then
	embDir=$2
fi

echo ${oFile}
echo ${embDir}
rm -f ${oFile}
#git rev-parse --short HEAD  > ${oFile}
#(cd EloiZ; git rev-parse --short HEAD >> ../${oFile}; cd ..)
embs=$(find ${embDir} -type f -follow -print)
echo -e  "- Similarity (fallback)\n" >> ${oFile}
python3 eval_similarity.py ${embs} >> ${oFile}
echo -e  "\n- Concreteness\n" >> ${oFile}
python3 eval_concreteness.py ${embs} >> ${oFile}
echo -e  "\n- Feature Norms\n" >> ${oFile}
python3 eval_fnorms.py ${embs} >> ${oFile}
echo -e  "\n- Lexical entialment\n" >> ${oFile}
python3 eval_lexical_entailment.py ${embs} >> ${oFile}
exit 0
echo -e  "\n- Analogy (fallback)\n" >> ${oFile}
echo -e  "topk 10\n" >> ${oFile}
python3 eval_analogy.py --topk 10 ${embs} >> ${oFile}
echo -e  "topk 1\n" >> ${oFile}
python3 eval_analogy.py --topk 1 ${embs} >> ${oFile}
echo -e  "\n- Analogy (nofallback)\n" >> ${oFile}
echo -e  "topk 10\n" >> ${oFile}
python3 eval_analogy.py --nofallback --topk 10 ${embs} >> ${oFile}
echo -e  "topk 1\n" >> ${oFile}
python3 eval_analogy.py --nofallback  --topk 1 ${embs} >> ${oFile}
