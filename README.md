# Clojure Deep Learning By MXNet

### 终于等到你...

* https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package
* https://mxnet.incubator.apache.org/api/clojure/index.html
* https://medium.com/apache-mxnet/clojure-package-for-mxnet-799f15043809
* https://nextjournal.com/gigasquid/clojure-mxnet-introduction
* https://medium.com/magnetcoop/clojure-mxnet-for-musculoskeletal-disease-diagnosis-946f03b790a8
* http://gigasquidsoftware.com/blog/2018/07/05/clojure-mxnet-the-module-api/

```clojure
(defn predict [img-url show?]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/resnet-152") :epoch 0})
        labels (-> (slurp (str model-dir "/synset.txt"))
                   (string/split #"\n"))
        nd-img (get-image img-url show?)
        prob (-> mod
                 (m/bind {:for-training false 
                          :data-shapes [{:name "data" :shape [1 num-channels h w]}]})
                 (m/forward {:data [nd-img]})
                 (m/outputs)
                 (ffirst))
        prob-with-labels (mapv (fn [p l] {:prob p :label l})
                               (ndarray/->vec prob)
                               labels)]
    (->> (sort-by :prob prob-with-labels)
         (reverse)
         (take 5))))         
```
