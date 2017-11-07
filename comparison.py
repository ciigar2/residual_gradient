import tensorflow as tf
import gym

def generate(env, policy, num_episodes=500, max_length=10000):
    data = []

    for _ in range(num_episodes):
        state = env.reset()
        episode = []
        step = 0
        done = False

        while (not done) and (step < max_length):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            terminal = [0.0] if done else [1.0]
            episode.append({'state' : state,
                            'action' : action,
                            'reward' : [reward],
                            'next_state' : next_state,
                            'terminal' : terminal})
            state = next_state
            step += 1

        data.append(episode)

    return data


def comparison(model, training_episodes, test_episodes, gamma=1.0, alpha=0.001, iterations=1000):

    # Initialize session
    sess = tf.Session()

    # monte carlo estimates
    def process(episodes):
        states = []
        rewards = []
        next_states = []
        terminals = []
        totals = []

        for episode in episodes:
            for i in range(len(episode)):
                states.append(episode[i]['state'])
                rewards.append(episode[i]['reward'])
                next_states.append(episode[i]['next_state'])
                terminals.append(episode[i]['terminal'])

                total = 0
                discount = 1.0

                for j in range(i, len(episode)):
                    total += discount * episode[j]['reward'][0]
                    discount *= gamma

                totals.append([total])

        return states, rewards, next_states, terminals, totals

    train_states, train_rewards, train_next, train_terminals, train_totals = process(training_episodes)
    test_states, test_rewards, test_next, test_terminals, test_totals = process(test_episodes)

    # return if no training data provided
    if len(train_states) == 0:
        return

    # Build loss functions
    size = train_states[0].shape[0]

    state = tf.placeholder(tf.float32, [None, size], 'state')
    next_state = tf.placeholder(tf.float32, [None, size], 'next_state')

    reward = tf.placeholder(tf.float32, [None, 1], 'reward')
    terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
    total = tf.placeholder(tf.float32, [None, 1], 'total')

    value = model(state)
    next_value = model(next_state)

    residual_loss = tf.reduce_mean(tf.square(reward + gamma * tf.multiply(terminal, next_value) - value))
    td_loss = tf.reduce_mean(tf.square(reward + gamma * tf.stop_gradient(tf.multiply(terminal, next_value)) - value))
    monte_carlo_loss = tf.reduce_mean(tf.square(total - value))

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha)

    residual_optimizer = optimizer.minimize(residual_loss)
    td_optimizer = optimizer.minimize(td_loss)
    monte_carlo_optimizer = optimizer.minimize(monte_carlo_loss)

    # Set up feed dictionaries
    train_dict = {state:train_states, reward:train_rewards, next_state:train_next, terminal:train_terminals, total:train_totals}
    test_dict = {state:test_states, reward:test_rewards, next_state:test_next, terminal:test_terminals, total:test_totals}

    # Evaluate residual gradient
    sess.run(tf.global_variables_initializer())

    for iteration in range(iterations):
        print('residual, iteration: ', iteration)
        sess.run(residual_optimizer, feed_dict=train_dict)

    residual_loss_train = sess.run(residual_loss, feed_dict=train_dict)
    residual_loss_test = sess.run(residual_loss, feed_dict=test_dict)
    residual_mc_train = sess.run(monte_carlo_loss, feed_dict=train_dict)
    residual_mc_test = sess.run(monte_carlo_loss, feed_dict=test_dict)

    # Evaluate TD
    sess.run(tf.global_variables_initializer())

    for iteration in range(iterations):
        print('TD, iteration: ', iteration)
        sess.run(td_optimizer, feed_dict=train_dict)

    td_loss_train = sess.run(td_loss, feed_dict=train_dict)
    td_loss_test = sess.run(td_loss, feed_dict=test_dict)
    td_mc_train = sess.run(monte_carlo_loss, feed_dict=train_dict)
    td_mc_test = sess.run(monte_carlo_loss, feed_dict=test_dict)

    # Evaluate monte carlo
    sess.run(tf.global_variables_initializer())

    for iteration in range(iterations):
        print('monte carlo, iteration: ', iteration)
        sess.run(monte_carlo_optimizer, feed_dict=train_dict)

    mc_train = sess.run(monte_carlo_loss, feed_dict=train_dict)
    mc_test = sess.run(monte_carlo_loss, feed_dict=test_dict)

    print('residual test error: ', residual_loss_test, ', residual train error: ', residual_loss_train)
    print('td test error: ', td_loss_test, ', td train error: ', td_loss_train)

    print('test MC err: ', residual_mc_test, ', train MC err: ', residual_mc_train, ' -- residual gradient')
    print('test MC err: ', td_mc_test, ', train MC err: ', td_mc_train, ' -- TD')
    print('test MC err: ', mc_test, ', train MC err: ', mc_train, ' -- monte carlo')


def mpl(input_size, hidden_layers=1, hidden_nodes=10):
    input_weights = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.35))
    input_biases = tf.Variable(tf.ones([hidden_nodes]))
    output_weights = tf.Variable(tf.random_normal([hidden_nodes, 1], stddev=0.35))
    output_bias = tf.Variable(tf.ones([1]))

    weights = []
    biases = []

    for _ in range(hidden_layers - 1):
        weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], stddev=0.35)))
        biases.append(tf.Variable(tf.ones([hidden_nodes])))

    def output(input):
        value = tf.nn.relu(tf.add(tf.matmul(input, input_weights), input_biases))

        for l in range(hidden_layers - 1):
            value = tf.nn.relu(tf.add(tf.matmul(value, weights[l]), biases[l]))

        return tf.add(tf.matmul(value, output_weights), output_bias)

    return output


def mountainCar():
    print('Starting experiment in mountain car domain')
    env = gym.make('MountainCar-v0')

    def policy(state):
        if state[1] > 0:
            return 2
        else:
            return 0

    train = generate(env, policy, num_episodes=100)
    test = generate(env, policy, num_episodes=50)

    comparison(mpl(env.observation_space.shape[0], hidden_layers=1, hidden_nodes=100),
               train, test, gamma=0.95, alpha=0.001, iterations=5000)

mountainCar()



