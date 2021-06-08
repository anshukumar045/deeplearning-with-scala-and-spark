package com.kanshu.ScalaFP.deeplearning

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

object DeepNet {

  import HyperParameters._

  val denseLayer = new DenseLayer.Builder()
    .activation(Activation.RELU)
    .nIn(nInputs)
    .nOut(nHiddenNodes)
    .weightInit(WeightInit.RELU).build()

  val outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .activation(Activation.SOFTMAX)
    .nIn(nHiddenNodes)
    .nOut(nOutputs)
    .build()

  val multiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(new Nesterovs(learningRate, 0.9))
    .list()
    .layer(0, denseLayer)
    .layer(1, outputLayer)
    .build()


}
