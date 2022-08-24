import cohere
import re

co = cohere.Client('QFTD2O4Y78t2iGFuy1bBYuqS4d3NrNi84KUww6wW')

def co_request(prompt, gens, description):
    response = co.generate(
      model='xlarge',
      prompt= prompt + ' ' + description + '. Some example phrases are:',
      max_tokens=80,
      temperature=1,
      k=0,
      p=0.7,
      frequency_penalty=0.04,
      presence_penalty=0,
      stop_sequences=["--"],
      return_likelihoods='NONE',
      num_generations=gens)
    return response

def generate_examples(prompt, num_examples, description, timeout=20, route=None):
    output = []
    calls = 0
    while len(output) < num_examples:
        if calls >= timeout:
            return {'message': "timeout error: too many calls made. Returned results up until failure", 'results': output}
            break
        response = co_request(prompt, 1, description)
        calls += 1
        for gen in response.generations:
            examples = [re.sub('\d[.]', '', g) for g in gen.text.split('\n')]
            for e in examples:
                if len(output) == num_examples:
                    break
                if not e.strip() in output and not e == '--' and not e == '' and len(e.split()) > 2:
                    output.append(e.strip())
    print('generated', len(output), 'for', route)
    return {'message': "Results successfully created", 'results': output}