(= (total $c0 $c1 $c2) (+ $c0 (+ $c1 $c2)))



(= (probablity $c $n) (/ $c $n))



(= (gini-impurity $c0 $c1 $c2)
   (let $n (total $c0 $c1 $c2)
        (- 1
           (+ (* (probablity $c0 $n) (probablity $c0 $n))
              (+ (* (probablity $c1 $n) (probablity $c1 $n))
                 (* (probablity $c2 $n) (probablity $c2 $n)))))))



(= (log2 $x) (log-math 2 $x))



(= (entropy-term $p) (if (> $p 0) (* (- 0 $p) (log2 $p)) 0))


(= (entropy-impurity $c0 $c1 $c2)
   (let $n (total $c0 $c1 $c2)
        (+ (entropy-term (probablity $c0 $n))
           (+ (entropy-term (probablity $c1 $n))
              (entropy-term (probablity $c2 $n))))))


(= (max3 $a $b $c) (max $a (max $b $c)))



(= (accuracy-score $c0 $c1 $c2)
   (/ (max3 $c0 $c1 $c2) (total $c0 $c1 $c2)))



(= (misclassification-rate $c0 $c1 $c2)
   (- 1 (accuracy-score $c0 $c1 $c2)))



(= (weighted-average $leftVal $rightVal $leftN $rightN)
   (/ (+ (* $leftN $leftVal) (* $rightN $rightVal)) (+ $leftN $rightN)))
