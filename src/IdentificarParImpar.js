const Rna = require('./lib/Rna');
const rl = require('readline-sync');

async function main() {
  const rna = new Rna({ arredondarResultados: true, funcaoAtivacao: 'sigmoid', passoDeTreinamento: 0.01, qtdeEntradas: 64, funcaoConversaoEntrada: intToBin });
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
    10: 0
  };
  await rna.treinarAte({ erroSerMenorQue: 0.01, log: true, valores: valoresExemplo });
  console.log("RNA treinada.");
  console.log("Testes: ");
  rna.imprimirTeste([9, 10, 11, 12, 13, 14]);
  while (true) {
    const entradaUsuario = rl.question('Entre com um valor inteiro: ');
    const numero = parseInt(entradaUsuario);
    console.log("Resultado", rna.predizer(numero));
  }
}

function intToBin(num) {
  let result = (num >>> 0).toString(2);
  while (result.length < 64) {
    result = "0" + result;
  }
  return result.split("").map(v => parseInt(v));
}

main().then();
