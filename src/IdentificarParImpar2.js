const Rna = require('./lib/Rna2');
const rl = require('readline-sync');

require('@tensorflow/tfjs-node');

async function main() {
  const rna = new Rna({ arredondarResultados: false, funcaoAtivacao: 'sigmoid', passoDeTreinamento: 0.001 });
  console.log('Treinando a RNA para identificar numeros pares ou impares ...');
  const valoresExemplo = {
    1: 1,
    2: 0,
    3: 1,
    4: 0,
    5: 1,
    6: 0,
    7: 1,
    8: 0,
    9: 1,
    10: 0,
    11: 1,
    12: 0,
    13: 1,
    14: 0,
    15: 1,
    16: 0,
    17: 1,
    18: 0,
    19: 1,
    20: 0
  };
  //await rna.treinarAte({ erroSerMenorQue: 0.01, log: false, valores: valoresExemplo });
  await rna.treinar(valoresExemplo, 50000);
  console.log("RNA treinada.");
  console.log("Testes: ");
  rna.imprimirTeste([9, 10, 11, 12, 13, 14]);
  while (true) {
    const entradaUsuario = rl.question('Entre com um valor inteiro: ');
    const numero = parseInt(entradaUsuario);
    console.log("Resultado", rna.predizer(numero));
  }
}

main().then();
