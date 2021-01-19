const tf = require('@tensorflow/tfjs');

class Rna {

  constructor({ funcaoAtivacao, arredondarResultados, passoDeTreinamento, funcaoConversaoEntrada, qtdeEntradas = 1 }) {
    this.arredondarResultados = arredondarResultados;
    this.funcaoConversaoEntrada = funcaoConversaoEntrada;
    this.model = tf.sequential();
    if (funcaoAtivacao) {
      this.model.add(tf.layers.dense({ units: 1, inputShape: [qtdeEntradas], activation: funcaoAtivacao }));
    } else {
      this.model.add(tf.layers.dense({ units: 1, inputShape: [qtdeEntradas] }));
    }
    if (passoDeTreinamento) {
      this.model.compile({ loss: 'meanSquaredError', optimizer: tf.train.rmsprop(passoDeTreinamento) });
    } else {
      this.model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
    }
  }

  async treinar(valores) {
    const valoresSeparados = this.separarValores(valores);
    let entradas;
    if (valoresSeparados.entradas[0] instanceof Array) {
      entradas = tf.tensor2d(valoresSeparados.entradas);
    } else {
      entradas = tf.tensor1d(valoresSeparados.entradas);
    }
    const saidas = tf.tensor1d(valoresSeparados.saidas);
    const resultado = await this.model.fit(entradas, saidas);
    return { erro: resultado.history.loss[0] };
  }

  async treinarAte({ erroSerMenorQue, log, valores }) {
    const epocas = 10000;
    let ultimoErro = 0;
    for (let i = 0; i < epocas; i++) {
      const resultado = await this.treinar(valores);
      if (log) {
        console.log('Erro:', resultado.erro);
      }
      if (resultado.erro < erroSerMenorQue || ultimoErro === resultado.erro) {
        break;
      }
      ultimoErro = resultado.erro;
    }
  }

  predizer(valor) {
    const valorConvertido = this.funcaoConversaoEntrada ? this.funcaoConversaoEntrada(valor) : valor;
    let entrada;
    if (valorConvertido instanceof Array) {
      entrada = tf.tensor2d([valorConvertido]);
    } else {
      entrada = tf.tensor1d([valorConvertido]);
    }
    const saida = this.model.predict(entrada).arraySync();
    if (this.arredondarResultados) {
      return Math.round(saida[0]);
    }
    return saida[0];
  }

  separarValores(valores) {
    let entradas = Object.keys(valores);
    const saidas = entradas.map(e => valores[e]);
    entradas = entradas.map(e => parseInt(e));
    if (this.funcaoConversaoEntrada) {
      entradas = entradas.map(e => this.funcaoConversaoEntrada(e));
    }
    return { entradas, saidas };
  }

  imprimirTeste(valores) {
    for (const valor of valores) {
      const resultado = this.predizer(valor);
      console.log(`${valor}: ${resultado}`);
    }
  }

}

module.exports = Rna;