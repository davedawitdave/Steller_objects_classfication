(= (gini-gain $p0 $p1 $p2 $l0 $l1 $l2 $r0 $r1 $r2)
   (- (gini-impurity $p0 $p1 $p2)
      (weighted-average (gini-impurity $l0 $l1 $l2)
                         (gini-impurity $r0 $r1 $r2)
                         (total $l0 $l1 $l2)
                         (total $r0 $r1 $r2))))


(= (entropy-gain $p0 $p1 $p2 $l0 $l1 $l2 $r0 $r1 $r2)
   (- (entropy-impurity $p0 $p1 $p2)
      (weighted-average (entropy-impurity $l0 $l1 $l2)
                         (entropy-impurity $r0 $r1 $r2)
                         (total $l0 $l1 $l2)
                         (total $r0 $r1 $r2))))



(= (accuracy-gain $p0 $p1 $p2 $l0 $l1 $l2 $r0 $r1 $r2)
   (- (misclassification-rate $p0 $p1 $p2)
      (weighted-average (misclassification-rate $l0 $l1 $l2)
                         (misclassification-rate $r0 $r1 $r2)
                         (total $l0 $l1 $l2)
                         (total $r0 $r1 $r2))))
