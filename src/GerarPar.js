const Rna = require('./lib/Rna');
const rl = require('readline-sync');

async function main() {
  const rna = new Rna({ arredondarResultados: true });
  console.log('Treinando a RNA para gerar numeros pares ...');
  const valoresExemplo = {
    1: 2,
    2: 4,
    3: 6,
    4: 8
  };
  await rna.treinarAte({ erroSerMenorQue: 0.001, log: true, valores: valoresExemplo });
  console.log("RNA treinada.");
  console.log("Testes: ");
  rna.imprimirTeste([5, 6, 7]);
  while (true) {
    const entradaUsuario = rl.question('Entre com um valor para gerar numero par: ');
    const numero = parseInt(entradaUsuario);
    console.log("Resultado", rna.predizer(numero));
  }
}

main().then();
