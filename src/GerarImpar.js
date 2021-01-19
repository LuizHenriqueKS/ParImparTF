const Rna = require('./lib/Rna');
const rl = require('readline-sync');

async function main() {
  const rna = new Rna({ arredondarResultados: true });
  console.log('Treinando a RNA para gerar numeros impares ...');
  const valoresExemplo = {
    1: 1,
    2: 3,
    3: 5,
    4: 7
  };
  await rna.treinarAte({ erroSerMenorQue: 0.001, log: true, valores: valoresExemplo });
  console.log("RNA treinada.");
  console.log("Testes: ");
  rna.imprimirTeste([5, 6, 7]);
  while (true) {
    const entradaUsuario = rl.question('Entre com um valor para gerar numero impar: ');
    const numero = parseInt(entradaUsuario);
    console.log("Resultado", rna.predizer(numero));
  }
}

main().then();
